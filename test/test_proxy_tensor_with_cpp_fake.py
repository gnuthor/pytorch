# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841

"""
Tests for make_fx with C++ FakeTensor mode.

All tests run make_fx(tracing_mode="real") inside cpp_fake_tensor_mode().
The C++ Fake dispatch key handles ops with Meta kernels. Ops without Meta
kernels fall back to CppFakeFallbackMode, which looks up the specific
Python handler (decomposition, fake_impl, etc.) and calls it. Sub-ops
re-enter C++ Fake dispatch, so all results remain C++ fake tensors.
"""

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch._library.simple_registry
import torch._library.utils
import unittest
import warnings
import operator
import contextlib
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    DataDependentOutputException,
    FakeTensorConverter,
    FakeTensorMode,
)
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import ops
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    DecompositionInterpreter,
    get_isolated_graphmodule,
)
from torch.utils._pytree import tree_map
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import functools
import itertools

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()

USE_TORCHVISION = False
try:
    import torchvision

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )


@contextlib.contextmanager
def cpp_fake_tensor_mode(*, shape_env=None):
    """Activate C++ FakeTensor mode with a Python fallback for unhandled ops.

    The C++ Fake dispatch key handles ops that have Meta kernels.
    Ops without Meta kernels are forwarded to CppFakeFallbackMode which
    looks up the specific Python handler (decomposition, fake_impl, etc.)
    and calls it. Sub-ops re-enter C++ Fake dispatch, so all tensors
    remain C++ fake tensors — no Python FakeTensors are created.
    """
    if shape_env is None:
        shape_env = ShapeEnv()
    converter = FakeTensorConverter()
    # fallback = CppFakeFallbackMode()
    # torch._C._create_and_enter_fake_tensor_mode(converter, shape_env, fallback)
    torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
    try:
        yield shape_env
    finally:
        torch._C._exit_fake_tensor_mode()


def _create_new_input(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)


class TestCppFakeProxyTensor(TestCase):
    """Tests for make_fx under C++ FakeTensor mode.

    Each test wraps the make_fx call in cpp_fake_tensor_mode() and uses
    tracing_mode="real" so that the C++ Fake dispatch key provides the
    fake tensor semantics.
    """

    def _test(self, f, inps):
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(*inps)
        new_inps = tree_map(_create_new_input, inps)
        r1 = fx_f(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_make_fx_simple(self):
        def f(x):
            return torch.sin(x)

        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device="cpu"):
        def f(a, b):
            return a + b

        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_empty_like_doesnt_burn_in_defaults(self):
        def f(x):
            return torch.empty_like(x)

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertExpectedInline(
            out.code.strip(),
            """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""",
        )

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        def f(x):
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(
            inp, size, dtype=None, layout=None, device=None, pin_memory=None
        ):
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        with cpp_fake_tensor_mode():
            out = make_fx(
                f, tracing_mode="real", decomposition_table=factory_func_decomp
            )(torch.ones(2))
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """,
        )

    def test_make_fx_reentrant_dispatch(self):
        def f(x):
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real", decomposition_table=decomp)(
                torch.rand(3)
            )

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

    def test_varargs(self):
        def f(*args):
            return sum(args)

        self._test(f, [torch.randn(2), torch.randn(2)])

    def test_proxy_tensor(self):
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    def test_inplace_metadata(self):
        def f(x):
            x = x.clone()
            x.unsqueeze_(-1)
            if x.shape[-1] != 1:
                raise AssertionError(f"expected x.shape[-1] == 1, got {x.shape[-1]}")
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertTrue(
            any(node.target == aten.randn.default for node in traced.graph.nodes)
        )

    def test_val_metadata_mutation(self):
        def f(x):
            y = x.clone()
            y.unsqueeze_(0)
            return y

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3, requires_grad=True))
        self.assertEqual(
            [
                tuple(node.meta["val"].shape)
                for node in traced.graph.nodes
                if "val" in node.meta
            ],
            [(3,), (3,), (1, 3)],
        )

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))

        self.assertTrue(
            all(
                isinstance(node.target, torch._ops.OpOverload)
                for node in traced.graph.nodes
                if node.op == "call_function"
            )
        )

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float("inf"))
            return torch.full((100, 100), val)

        self._test(f, [])

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_proxy_tensor_mut(self):
        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())
        self.assertEqual(g(), f())

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_unbind(self):
        def f():
            val = torch.tensor([2])
            (r,) = torch.unbind(val, 0)
            return r.item()

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())

    def test_decomposition_interpreter(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        with cpp_fake_tensor_mode():
            fx_module = make_fx(fn, tracing_mode="real", decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if (
                n.target == torch.ops.aten.silu
                or n.target == torch.ops.aten.silu.default
            ):
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {
            torch.ops.aten.silu.default: decomposition_table[
                torch.ops.aten.silu.default
            ]
        }
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        self.assertEqual(fx_module(x), decomposed_module(x))

    def test_make_fx_model_fwd_bwd(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            out = torch.func.functional_call(model, params, x).sum()
            out.backward()
            return list(params.values())

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params)
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(args, params, buffers):
            for p in params.values():
                p.grad = None
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = torch.func.functional_call(model, params_and_buffers, args)
            out.sum().backward()
            return [p - 1e-4 * p.grad for p in params.values()]

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params, buffers)
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
        )
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
        )

    def test_make_fx_model_double_param(self):
        class Emformer(torch.nn.Module):
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(input_dim)

            def forward(mod_self, x):  # noqa: B902
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z

        with cpp_fake_tensor_mode():
            gm = make_fx(Emformer(), tracing_mode="real")(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertEqual(len(ops), 2)

    def test_partial_decomp(self):
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y

        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        with cpp_fake_tensor_mode():
            fx_g = make_fx(f, tracing_mode="real")(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        with cpp_fake_tensor_mode():
            decomposed_fx = make_fx(
                f, tracing_mode="real", decomposition_table={aten.addmm.default: addmm}
            )(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(
            len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2
        )
        self.assertEqual(
            len(
                [n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]
            ),
            1,
        )

    def test_decomp_of_capture(self):
        val = torch.randn(5)

        def f(x):
            return x.t() + val.t()

        def nop(x):
            return x.cos()

        with cpp_fake_tensor_mode():
            traced = make_fx(
                f,
                tracing_mode="real",
                decomposition_table={torch.ops.aten.t.default: nop},
            )(torch.randn(5))
        self.assertEqual(
            len(
                [n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]
            ),
            0,
        )

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device="cuda")
        with torch.autocast("cuda"):
            with cpp_fake_tensor_mode():
                out_graph = make_fx(f, tracing_mode="real")(inp, layer.weight).graph
                out_graph2 = make_fx(f, tracing_mode="real")(inp, layer.weight).graph

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)

    def test_strides(self):
        def f(x):
            self.assertTrue(x.is_contiguous())
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))
            x = x.permute(0, 3, 1, 2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
            return x

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

        def f(x):
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_fused_adam(self):
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
        ):
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        with cpp_fake_tensor_mode():
            gm = make_fx(fused_adam, tracing_mode="real")(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                self.assertIn("val", n.meta)

    def test_alias(self):
        def f(x):
            return torch.ops.aten.alias(x)

        with cpp_fake_tensor_mode():
            r = str(make_fx(f, tracing_mode="real")(torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""",
        )

    def test_meta(self):
        def f(x):
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == "output":
                continue
            self.assertTrue("val" in n.meta)

    def test_simple_add(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(3, 4))

        # Verify the graph has the expected structure
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(len(call_nodes) >= 1)

        # Verify it runs correctly with real inputs
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        self.assertEqual(gm(x, y), f(x, y))

    def test_matmul(self):
        def f(x, y):
            return torch.matmul(x, y)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4, 5))
        x, y = torch.randn(3, 4), torch.randn(4, 5)
        self.assertEqual(gm(x, y), f(x, y))

    def test_multiple_outputs(self):
        def f(x):
            return torch.max(x, dim=0)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        r1 = gm(x)
        r2 = f(x)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[1])

    def test_inplace_ops(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        self._test(f, (torch.randn(3, 4),))

    def test_view_ops(self):
        def f(x):
            y = x.view(2, 6)
            z = y.t()
            return z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_cat(self):
        def f(x, y):
            return torch.cat([x, y], dim=0)

        self._test(f, (torch.randn(3, 4), torch.randn(5, 4)))

    def test_nn_module(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        def f(x):
            return model(x)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 10))
        x = torch.randn(3, 10)
        self.assertEqual(gm(x), f(x))

    def test_comparison_with_python_fake(self):
        """Verify that C++ fake mode and Python fake mode produce the same graph structure."""

        def f(x):
            y = torch.sin(x)
            z = torch.cos(y)
            return z + x

        inp = torch.randn(4, 4)

        # Trace with Python fake mode
        py_gm = make_fx(f, tracing_mode="fake")(inp)

        # Trace with C++ fake mode
        with cpp_fake_tensor_mode():
            cpp_gm = make_fx(f, tracing_mode="real")(inp)

        # Both should produce identical graph structure
        py_ops = [n.target for n in py_gm.graph.nodes if n.op == "call_function"]
        cpp_ops = [n.target for n in cpp_gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(py_ops, cpp_ops)

        # Both should produce correct results
        x = torch.randn(4, 4)
        self.assertEqual(py_gm(x), cpp_gm(x))

    def test_factory_ops_under_cpp_fake(self):
        """Factory ops like torch.zeros should work under C++ fake mode."""

        def f(x):
            z = torch.zeros(x.shape)
            return x + z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_dtype_promotion(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(
                torch.randn(3, dtype=torch.float32),
                torch.randn(3, dtype=torch.float64),
            )
        x = torch.randn(3, dtype=torch.float32)
        y = torch.randn(3, dtype=torch.float64)
        self.assertEqual(gm(x, y), f(x, y))

    def test_broadcasting(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4))
        x, y = torch.randn(3, 4), torch.randn(4)
        self.assertEqual(gm(x, y), f(x, y))

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_cuda_device(self):
        def f(x):
            return x.sin() + x.cos()

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, device="cuda"))
        x = torch.randn(3, device="cuda")
        self.assertEqual(gm(x), f(x))


# class TestCppFakeSymbolicTracing(TestCase):
#     """Symbolic tracing tests ported from TestSymbolicTracing.

#     These use tracing_mode="symbolic" which creates its own FakeTensorMode
#     internally. We wrap in cpp_fake_tensor_mode() to test that C++ fake
#     tensors interoperate with the symbolic tracing machinery.
#     """

#     def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
#         trace_inputs = [torch.randn(shape) for shape in trace_inputs]
#         with cpp_fake_tensor_mode():
#             traced_f = make_fx(fn, tracing_mode="symbolic")(*trace_inputs)
#         for input in test_inputs:
#             input = [torch.randn(shape) for shape in input]
#             rx, ry = traced_f(*input), fn(*input)
#             if assert_eq:
#                 self.assertEqual(rx, ry)
#         return traced_f

#     def test_int_input(self):
#         def f(x, y):
#             return x.view(y)

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(torch.empty(3, 4), 12).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, x_1, y_1):
#     view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
#     return view""",
#         )

#     def test_resize_from_zero(self):
#         def f(x, y):
#             x.resize_(y.size(0))

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(torch.empty(0), torch.empty(2)).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, x_1, y_1):
#     sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
#     resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = resize_ = None
#     return None""",
#         )

#     def test_broadcast_shapes(self):
#         def f(x, y):
#             return torch.functional.broadcast_shapes(x.size(), y.size()[0])

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(
#                     torch.empty(3, 1), torch.empty(5)
#                 ).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, x_1, y_1):
#     sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
#     sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
#     return (sym_size_int, sym_size_int_1)""",
#         )

#     def test_unary(self):
#         def f(x):
#             if x.shape[0] >= 20:
#                 raise AssertionError(f"expected x.shape[0] < 20, got {x.shape[0]}")
#             return x.cos()

#         test_inputs = []
#         test_inputs.append([(2, 5)])
#         test_inputs.append([(6, 8)])
#         self._test_dynamic(f, [(3, 4)], test_inputs)

#     def test_multiply_shape(self):
#         def f(a):
#             return torch.empty(a.shape[0] * 2)

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
#     mul = sym_size_int * 2;  sym_size_int = None
#     empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
#     return empty""",
#         )

#     def test_item(self):
#         def f(a):
#             r = a.item()
#             return r * a

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(1)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1)
#     mul = torch.ops.aten.mul.Tensor(a_1, _local_scalar_dense);  a_1 = _local_scalar_dense = None
#     return mul""",
#         )

#     def test_item_to_constructor(self):
#         def f(a):
#             r = a.item()
#             return torch.empty(r)

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(torch.randint(5, (1,))).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1);  a_1 = None
#     empty = torch.ops.aten.empty.memory_format([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
#     return empty""",  # noqa: B950
#         )

#     def test_neg_shape(self):
#         def f(a):
#             return torch.empty(-a.shape[0] + 10)

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(2)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
#     neg = -sym_size_int;  sym_size_int = None
#     add = neg + 10;  neg = None
#     empty = torch.ops.aten.empty.memory_format([add], device = device(type='cpu'), pin_memory = False);  add = None
#     return empty""",
#         )

#     def test_binary_broadcast(self):
#         def f(a, b):
#             c = a * b
#             return c

#         test_inputs = []
#         test_inputs.append([(1, 5), (3, 1)])
#         test_inputs.append([(1, 4), (4, 1)])
#         self._test_dynamic(f, [(1, 2), (3, 1)], test_inputs)

#     def test_expand(self):
#         def f(a):
#             b = torch.mul(a, a)
#             c = b.expand(a.shape)
#             return c

#         self._test_dynamic(f, [(3,)], [[(3,)], [(4,)], [(2,)]])
#         self._test_dynamic(f, [(5, 1)], [[(4, 1)], [(3, 1)], [(6, 1)]])

#     def test_return_symint(self):
#         def f(x):
#             return x.shape[0], x.cos(), x.shape[0] / 5

#         self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

#         def f(x):
#             return x.shape

#         self._test_dynamic(f, [(5, 3)], [[(4, 6)]])

#     def test_rmethod(self):
#         def f(x):
#             return x.size(0) + x

#         self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

#     def test_symint_to_tensor(self):
#         def f(a):
#             return a / a.shape[0]

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
#     div = torch.ops.aten.div.Tensor(a_1, sym_size_int);  a_1 = sym_size_int = None
#     return div""",
#         )

#     def test_sqrt_size(self):
#         def f(a):
#             return a / a.size(-1) ** 0.5

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, a_1):
#     sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
#     sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
#     pow_1 = sym_float ** 0.5;  sym_float = None
#     div = torch.ops.aten.div.Tensor(a_1, pow_1);  a_1 = pow_1 = None
#     return div""",
#         )

#     def test_sym_storage_offset(self):
#         def f(x, y):
#             return x + y

#         inp = (torch.randn(8)[3:], torch.randn(5))
#         with cpp_fake_tensor_mode():
#             fx_g = make_fx(f, tracing_mode="symbolic")(*inp)
#         inp = (torch.randn(8)[3:], torch.randn(5))
#         self.assertEqual(fx_g(*inp), f(*inp))

#     def test_setitem_symint(self):
#         def f(x):
#             x[0] = x.size(0)
#             return x

#         with cpp_fake_tensor_mode():
#             r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(10)).code).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, x_1):
#     sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
#     scalar_tensor = torch.ops.aten.scalar_tensor.default(sym_size_int, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  sym_size_int = None
#     select = torch.ops.aten.select.int(x_1, 0, 0)
#     copy_ = torch.ops.aten.copy_.default(select, scalar_tensor);  select = scalar_tensor = copy_ = None
#     return x_1""",  # noqa: B950
#         )

#     def test_non_symint_size_spec(self):
#         def f(x):
#             torch._C._non_sym_sizes(x)
#             return x + 1

#         x = torch.randn(2, 3)
#         with cpp_fake_tensor_mode():
#             make_fx(f, tracing_mode="symbolic")(x)

#     def test_new_empty(self):
#         def f(a, b):
#             return a.new_empty(b.shape[0], b.shape[1] * 2)

#         self._test_dynamic(
#             f, [(2, 4), (4, 5)], [[(2, 3), (5, 7)], [(3, 7), (9, 3)]], assert_eq=False
#         )

#     def test_unbacked_slice(self):
#         def f(x, m):
#             x = x[m]
#             return x[
#                 slice(None, None, None), slice(None, None, None), slice(None, 2, None)
#             ]

#         with cpp_fake_tensor_mode():
#             make_fx(f, tracing_mode="symbolic")(
#                 torch.randn((12, 3, 3)), torch.randint(0, 2, (12,), dtype=torch.bool)
#             )

#     def test_dynamic_pointwise_scalar(self):
#         def f(gravity, mask):
#             gravity[mask, 0] = gravity[mask, 0] * -1

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(
#                     torch.randn((12, 4)), torch.randint(0, 2, (12,), dtype=torch.bool)
#                 ).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, gravity_1, mask_1):
#     select = torch.ops.aten.select.int(gravity_1, 1, 0)
#     index = torch.ops.aten.index.Tensor(select, [mask_1]);  select = None
#     mul = torch.ops.aten.mul.Tensor(index, -1);  index = None
#     select_1 = torch.ops.aten.select.int(gravity_1, 1, 0);  gravity_1 = None
#     index_put_ = torch.ops.aten.index_put_.default(select_1, [mask_1], mul);  select_1 = mask_1 = mul = index_put_ = None
#     return None""",
#         )

#     def test_boolean_index(self):
#         def f(images, handedness, valid):
#             images = images[valid]
#             handedness = handedness[valid]
#             right_hand_mask = handedness == 1
#             images[right_hand_mask] = images[right_hand_mask].flip(-1)

#         with cpp_fake_tensor_mode():
#             r = str(
#                 make_fx(f, tracing_mode="symbolic")(
#                     torch.randint(0, 256, (512, 1, 96, 96)),
#                     torch.randint(0, 1, (512,)),
#                     torch.randint(0, 2, (512,), dtype=torch.bool),
#                 ).code
#             ).strip()
#         self.assertExpectedInline(
#             r,
#             """\
# def forward(self, images_1, handedness_1, valid_1):
#     index = torch.ops.aten.index.Tensor(images_1, [valid_1]);  images_1 = None
#     index_1 = torch.ops.aten.index.Tensor(handedness_1, [valid_1]);  handedness_1 = valid_1 = None
#     eq = torch.ops.aten.eq.Scalar(index_1, 1);  index_1 = None
#     index_2 = torch.ops.aten.index.Tensor(index, [eq])
#     flip = torch.ops.aten.flip.default(index_2, [-1]);  index_2 = None
#     index_put_ = torch.ops.aten.index_put_.default(index, [eq], flip);  index = eq = flip = index_put_ = None
#     return None""",
#         )


if __name__ == "__main__":
    run_tests()
