# Add -rpath-link dirs so the conda compat linker (-B compiler_compat) can
# find versioned shared libraries (libnuma.so.1, libgomp.so.1,
# libroctracer64.so.4, ...) when resolving transitive DT_NEEDED during
# test-binary and torch_shm_manager linking.
# Must be done here (before add_subdirectory) so subdirectory scopes inherit
# the flags. System lib dirs come first to prefer system-native libraries
# over conda's sysroot versions (which trigger GLIBC_PRIVATE link failures).
# -Wl,-rpath-link is a GNU ld / ELF-only concept; skip on Windows and macOS.
#
# Requires _env_prefix to be set (list of CMAKE_PREFIX_PATH entries from env).

set(_rp_flags "")
if(NOT WIN32 AND NOT APPLE)
# System multiarch lib dir (Debian/Ubuntu: /lib/x86_64-linux-gnu, etc.).
if(CMAKE_LIBRARY_ARCHITECTURE)
  foreach(_d
      "/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
      "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    if(IS_DIRECTORY "${_d}")
      string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
    endif()
  endforeach()
endif()
# Traditional 64-bit lib dirs (RHEL/manylinux).
foreach(_d "/lib64" "/usr/lib64")
  if(IS_DIRECTORY "${_d}")
    string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
  endif()
endforeach()
# Intel oneAPI MKL SYCL (XPU): libmkl_sycl_lapack.so.5 and siblings require
# libmkl_core.so.2 from oneAPI MKL 2025.3+ which has new batch-strided LAPACK
# symbols (mkl_lapack_spotrf_batch_strided, etc.). Must appear before
# ENV{CMAKE_PREFIX_PATH} (conda) so the linker finds the oneAPI libmkl_core.so.2
# instead of conda's older version that lacks these symbols.
file(GLOB _mkl_lib_dirs "/opt/intel/oneapi/mkl/*/lib")
foreach(_d IN LISTS _mkl_lib_dirs)
  if(IS_DIRECTORY "${_d}")
    string(APPEND _rp_flags " -Wl,-rpath-link,${_d}")
  endif()
endforeach()
# ENV{CMAKE_PREFIX_PATH} entries (conda env, etc.).
foreach(_prefix IN LISTS _env_prefix)
  foreach(_sub "lib" "lib64")
    if(IS_DIRECTORY "${_prefix}/${_sub}")
      string(APPEND _rp_flags " -Wl,-rpath-link,${_prefix}/${_sub}")
    endif()
  endforeach()
endforeach()
# ROCm: use ROCM_PATH env var (set in ROCm CI), fall back to /opt/rocm.
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  set(_rocm_root "$ENV{ROCM_PATH}")
else()
  set(_rocm_root "/opt/rocm")
endif()
foreach(_sub "lib" "lib64")
  if(IS_DIRECTORY "${_rocm_root}/${_sub}")
    string(APPEND _rp_flags " -Wl,-rpath-link,${_rocm_root}/${_sub}")
  endif()
endforeach()
# CUDA: use CUDA_PATH env var, fall back to /usr/local/cuda.
if(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
  set(_cuda_root "$ENV{CUDA_PATH}")
else()
  set(_cuda_root "/usr/local/cuda")
endif()
foreach(_sub "lib64" "lib")
  if(IS_DIRECTORY "${_cuda_root}/${_sub}")
    string(APPEND _rp_flags " -Wl,-rpath-link,${_cuda_root}/${_sub}")
  endif()
endforeach()
# CUPTI (CUDA Profiling Tools Interface) lives in extras/CUPTI/lib64, not lib64.
if(IS_DIRECTORY "${_cuda_root}/extras/CUPTI/lib64")
  string(APPEND _rp_flags " -Wl,-rpath-link,${_cuda_root}/extras/CUPTI/lib64")
endif()
# MAGMA (ROCm): libmagma.so lives in MAGMA_HOME/lib, not in /opt/rocm/lib.
if(DEFINED ENV{MAGMA_HOME} AND NOT "$ENV{MAGMA_HOME}" STREQUAL "")
  string(APPEND _rp_flags " -Wl,-rpath-link,$ENV{MAGMA_HOME}/lib")
elseif(IS_DIRECTORY "${_rocm_root}/magma/lib")
  string(APPEND _rp_flags " -Wl,-rpath-link,${_rocm_root}/magma/lib")
endif()
# Aotriton (ROCm): installed into the source tree's torch/lib during the
# build phase. The linker ignores non-existent rpath-link dirs, so it is
# safe to add this unconditionally.
string(APPEND _rp_flags " -Wl,-rpath-link,${PROJECT_SOURCE_DIR}/torch/lib")
# torch-xpu-ops sycltla libs (XPU): libtorch-xpu-ops-sycltla-mha_*.so are
# built into the cmake binary dir's lib/ subdir during the build phase.
string(APPEND _rp_flags " -Wl,-rpath-link,${CMAKE_BINARY_DIR}/lib")
endif() # NOT WIN32 AND NOT APPLE
if(_rp_flags)
  string(APPEND CMAKE_EXE_LINKER_FLAGS "${_rp_flags}")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS "${_rp_flags}")
  set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}"    CACHE STRING "" FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "" FORCE)
endif()
