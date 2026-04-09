#!/usr/bin/env python3

import argparse
import csv
import functools
import time

import torch
from torch._inductor.autoheuristic.autoheuristic_utils import AHContext, AHMetadata
from torch._inductor.fx_passes.pad_mm import get_alignment_size_dtype
from torch._inductor.utils import get_gpu_shared_memory


def fits_in_memory(dtype, m: int, k: int, n: int) -> bool:
    threshold_memory = torch.cuda.get_device_properties(0).total_memory / 4
    return dtype.itemsize * (m * k + k * n + m * n) < threshold_memory


def set_precision(dtype) -> None:
    precision = "highest" if dtype == torch.float32 else "high"
    torch.set_float32_matmul_precision(precision)


def get_heuristic_decision(m: int, k: int, n: int, dtype: torch.dtype) -> str | None:
    from torch._inductor.autoheuristic.autoheuristic import AutoHeuristic, LocalFeedback
    from torch._inductor.fx_passes.pad_mm import (
        get_alignment_size,
        get_context,
        get_padded_length,
        pad_mm_operations,
        pad_mm_precondition,
    )

    torch._inductor.config.autoheuristic_use = "pad_mm"

    if not torch._inductor.config.run_autoheuristic("pad_mm"):
        return None

    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    m_padded_length = get_padded_length(m, get_alignment_size(a))
    k_padded_length = get_padded_length(k, get_alignment_size(a))
    n_padded_length = get_padded_length(n, get_alignment_size(b))

    context = get_context(
        a,
        b,
        mat1_pre_padded=False,
        mat2_pre_padded=False,
        m_padded_length=m_padded_length,
        k_padded_length=k_padded_length,
        n_padded_length=n_padded_length,
    )

    def dummy_feedback(choice: str) -> float:
        return 1.0

    def fallback() -> str:
        return "autotune"

    autoheuristic = AutoHeuristic(
        fallback=fallback,
        choices=["orig", "pad"],
        feedback=LocalFeedback(dummy_feedback),
        context=context,
        name="pad_mm",
        augment_context=pad_mm_operations(),
        precondition=pad_mm_precondition,
    )

    choice = autoheuristic.get_choice()
    # Return the actual choice made by the heuristic
    # "autotune" means fallback to benchmarking (heuristic not confident)
    return choice


def benchmark_both_choices(
    m: int, k: int, n: int, dtype: torch.dtype, num_reps: int = 3
) -> tuple[float, float]:
    set_precision(dtype)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    def benchmark_fn(fn, num_warmup=5):
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(num_reps):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        return sum(times) / len(times)

    orig_time = benchmark_fn(lambda: torch.mm(a, b))

    from torch._inductor.fx_passes.pad_mm import (
        get_alignment_size,
        get_padded_length,
        pad_mm,
    )

    m_padded_length = get_padded_length(a.shape[0], get_alignment_size(a))
    k_padded_length = get_padded_length(a.shape[1], get_alignment_size(a))
    n_padded_length = get_padded_length(b.shape[1], get_alignment_size(b))

    if m_padded_length == 0 and k_padded_length == 0 and n_padded_length == 0:
        return orig_time, orig_time

    def pad_fn():
        return pad_mm(a, b, m_padded_length, k_padded_length, n_padded_length)

    pad_time = benchmark_fn(pad_fn)
    return orig_time, pad_time


def load_shapes_from_csv(csv_file: str) -> list:
    shapes = []
    try:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                m, k, n = int(row["M"]), int(row["K"]), int(row["N"])
                dtype_str = row["dtype"]

                if dtype_str == "float16":
                    dtype = torch.float16
                elif dtype_str == "bfloat16":
                    dtype = torch.bfloat16
                elif dtype_str == "float32":
                    dtype = torch.float32
                else:
                    continue

                shapes.append((m, k, n, dtype))

        print(f"Loaded {len(shapes)} shapes from {csv_file}")
        return shapes
    except Exception as e:
        print(f"Error loading shapes from {csv_file}: {e}")
        return []


@functools.cache
def get_shared_mem_size():
    return get_gpu_shared_memory()


def check_shape_passes_precondition(m: int, k: int, n: int, dtype: torch.dtype) -> bool:
    """
    Check if a shape passes the same precondition used by the actual pad_mm AutoHeuristics.

    This uses the exact same pad_mm_precondition function that the AutoHeuristic system
    uses, avoiding hardcoded magic numbers by delegating to the source of truth.
    """
    from torch._inductor.autoheuristic.autoheuristic_utils import pad_mm_precondition

    shared_memory = get_shared_mem_size()
    device_capa = torch.cuda.get_device_capability()

    # Create the same metadata and context that AutoHeuristics uses
    metadata = AHMetadata(
        shared_memory=shared_memory,
        device_capa=device_capa,
        choices=["orig", "pad"],  # Required but not used for precondition check
        name="pad_mm",  # Required but not used for precondition check
    )

    context = AHContext()
    context.add_feature("m", m)
    context.add_feature("k", k)
    context.add_feature("n", n)

    # Use the actual pad_mm_precondition function - no hardcoded values!
    return pad_mm_precondition(metadata, context)


def filter_shapes(shapes: list) -> list:
    filtered = []
    aligned_count = 0
    precondition_failed_count = 0
    memory_count = 0

    for m, k, n, dtype in shapes:
        # Check if already aligned
        align_size = get_alignment_size_dtype(dtype)
        is_aligned = all((dim % align_size == 0) for dim in [m, k, n])

        if is_aligned:
            aligned_count += 1
            continue

        # Check if passes the actual precondition used by pad_mm AutoHeuristics
        if not check_shape_passes_precondition(m, k, n, dtype):
            precondition_failed_count += 1
            continue

        # Check if fits in memory
        if not fits_in_memory(dtype, m, k, n):
            memory_count += 1
            continue

        # This shape is suitable for evaluation
        filtered.append((m, k, n, dtype))

    print("Filtering results:")
    print(f"  Already aligned (skipped): {aligned_count}")
    print(f"  Failed pad_mm_precondition (skipped): {precondition_failed_count}")
    print(f"  Too large for memory (skipped): {memory_count}")
    print(f"  Suitable for evaluation: {len(filtered)}")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained AutoHeuristics for pad_mm optimization"
    )
    parser.add_argument("csv_file", help="Path to CSV file with M,K,N,dtype columns")
    parser.add_argument(
        "--num-reps", type=int, default=3, help="Benchmark repetitions (default: 3)"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="CUDA device (default: current)"
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=10000,
        help="Max shapes to test (default: 10000)",
    )

    args = parser.parse_args()

    torch.set_default_device("cuda")
    if args.device is not None:
        torch.cuda.set_device(args.device)

    print(f"Using CUDA device: {torch.cuda.current_device()}")
    print()

    shapes = load_shapes_from_csv(args.csv_file)
    if not shapes:
        print("No shapes found!")
        return

    shapes = filter_shapes(shapes)
    if not shapes:
        print("No suitable shapes found!")
        return

    if len(shapes) > args.max_shapes:
        shapes = shapes[: args.max_shapes]
        print(f"Limited to first {args.max_shapes} shapes")

    print(f"Evaluating {len(shapes)} shapes with {args.num_reps} reps each")
    print()

    total_decisions = 0
    correct_decisions = 0
    true_positives = 0  # Chose pad, should pad
    true_negatives = 0  # Chose orig, should orig
    false_positives = 0  # Chose pad, should orig
    false_negatives = 0  # Chose orig, should pad
    skipped_shapes = 0
    autotune_shapes = 0

    tp_speedups = []  # Speed-up percentages for true positives
    fp_slowdowns = []  # Speed-down percentages for false positives

    # Track non-confident decisions and confident decisions by dtype
    autotune_shape_list = []  # List of (M, K, N, dtype) where heuristic chose autotune
    confident_by_dtype = {}  # Count of confident decisions by dtype

    for i, (m, k, n, dtype) in enumerate(shapes, 1):
        try:
            print(f"Shape {i}/{len(shapes)}: M={m}, K={k}, N={n}, dtype={dtype}")

            heuristic_choice = get_heuristic_decision(m, k, n, dtype)
            print(f"  Heuristic: {heuristic_choice}")

            if heuristic_choice is None:
                print("  (Skipped - error)")
                skipped_shapes += 1
                continue

            orig_time, pad_time = benchmark_both_choices(m, k, n, dtype, args.num_reps)
            ground_truth = "pad" if pad_time < orig_time else "orig"

            print(f"  Times: orig={orig_time:.6f}s, pad={pad_time:.6f}s")
            print(f"  Ground truth: {ground_truth}")

            if heuristic_choice == "autotune":
                # Heuristic punted to benchmarking - this is correct behavior for small/uncertain shapes
                autotune_shapes += 1
                autotune_shape_list.append((m, k, n, dtype))
                print("  Heuristic chose to benchmark (conservative)")
            else:
                # Heuristic made a confident decision - evaluate accuracy
                total_decisions += 1
                # Track confident decisions by dtype
                dtype_str = str(dtype).replace("torch.", "")
                confident_by_dtype[dtype_str] = confident_by_dtype.get(dtype_str, 0) + 1
                if heuristic_choice == ground_truth:
                    correct_decisions += 1
                    print("  ✓ CORRECT")
                    if heuristic_choice == "pad":
                        true_positives += 1  # Correctly chose pad
                        # Calculate speed-up: (orig_time - pad_time) / orig_time * 100
                        speedup = (orig_time - pad_time) / orig_time * 100
                        tp_speedups.append(speedup)
                        print(f"    Speed-up: {speedup:.1f}%")
                    else:
                        true_negatives += 1  # Correctly chose orig
                else:
                    print("  ✗ WRONG")
                    if heuristic_choice == "pad" and ground_truth == "orig":
                        false_positives += 1
                        # Calculate speed-down: (pad_time - orig_time) / orig_time * 100
                        slowdown = (pad_time - orig_time) / orig_time * 100
                        fp_slowdowns.append(slowdown)
                        print(f"    Speed-down: {slowdown:.1f}%")
                    elif heuristic_choice == "orig" and ground_truth == "pad":
                        false_negatives += 1

            print(f"  Confidence Rate: {total_decisions}/{i}")
            if total_decisions > 0:
                accuracy = correct_decisions / total_decisions * 100
                tp_rate = true_positives / total_decisions * 100
                tn_rate = true_negatives / total_decisions * 100
                fp_rate = false_positives / total_decisions * 100
                fn_rate = false_negatives / total_decisions * 100

                # Compute average speedup/slowdown
                avg_tp_speedup = (
                    sum(tp_speedups) / len(tp_speedups) if tp_speedups else 0
                )
                avg_fp_slowdown = (
                    sum(fp_slowdowns) / len(fp_slowdowns) if fp_slowdowns else 0
                )

                print(
                    f"  Accuracy: {correct_decisions}/{total_decisions} ({accuracy:.1f}%) "
                    f"| TP: {tp_rate:.1f}% (avg speedup: {avg_tp_speedup:.1f}%) "
                    f"| TN: {tn_rate:.1f}% "
                    f"| FP: {fp_rate:.1f}% (avg slowdown: {avg_fp_slowdown:.1f}%)"
                    f"| FN: {fn_rate:.1f}%"
                )

        except Exception as e:
            print(f"  Error: {e}")
            skipped_shapes += 1

        print()

    print("=== FINAL RESULTS ===")
    print(f"Confident decisions: {total_decisions}")
    print(f"Autotune (punted to benchmarking): {autotune_shapes}")
    print(f"Skipped (errors): {skipped_shapes}")

    if total_decisions > 0:
        accuracy = correct_decisions / total_decisions * 100
        tp_rate = true_positives / total_decisions * 100
        tn_rate = true_negatives / total_decisions * 100
        fp_rate = false_positives / total_decisions * 100
        fn_rate = false_negatives / total_decisions * 100

        avg_tp_speedup = sum(tp_speedups) / len(tp_speedups) if tp_speedups else 0
        avg_fp_slowdown = sum(fp_slowdowns) / len(fp_slowdowns) if fp_slowdowns else 0

        print(
            f"\nConfident decision accuracy: {accuracy:.1f}% ({correct_decisions}/{total_decisions})"
        )

        if tp_speedups:
            print(
                f"True Positives (chose pad, should pad): {tp_rate:.1f}% ({true_positives}) "
                f"| Avg speed-up: {avg_tp_speedup:.1f}%"
            )
        else:
            print(
                f"True Positives (chose pad, should pad): {tp_rate:.1f}% ({true_positives})"
            )

        print(
            f"True Negatives (chose orig, should orig): {tn_rate:.1f}% ({true_negatives})"
        )

        if fp_slowdowns:
            print(
                f"False Positives (chose pad, should orig): {fp_rate:.1f}% ({false_positives}) "
                f"| Avg speed-down: {avg_fp_slowdown:.1f}%"
            )
        else:
            print(
                f"False Positives (chose pad, should orig): {fp_rate:.1f}% ({false_positives})"
            )

        print(
            f"False Negatives (chose orig, should pad): {fn_rate:.1f}% ({false_negatives})"
        )
    else:
        print("No confident decisions made!")

    total_evaluated = total_decisions + autotune_shapes
    if total_evaluated > 0:
        print(
            f"\nConfidence rate: ({total_decisions}/{total_evaluated} made confident decisions)"
        )

    # Print shapes where AutoHeuristics did not make a confident decision
    print(f"\n=== NON-CONFIDENT DECISIONS ({len(autotune_shape_list)}) ===")
    if autotune_shape_list:
        print("Shapes where AutoHeuristics chose 'autotune' (non-confident):")
        for m, k, n, dtype in autotune_shape_list:
            dtype_str = str(dtype).replace("torch.", "")
            print(f"  M={m}, K={k}, N={n}, dtype={dtype_str}")
    else:
        print("All shapes had confident decisions!")

    # Print confident decisions by dtype
    print("\n=== CONFIDENT DECISIONS BY DTYPE ===")
    if confident_by_dtype:
        print("Number of confident decisions per dtype:")
        for dtype_str, count in sorted(confident_by_dtype.items()):
            print(f"  {dtype_str}: {count} confident decisions")
        print(f"Total confident decisions: {sum(confident_by_dtype.values())}")
    else:
        print("No confident decisions made!")


if __name__ == "__main__":
    main()
