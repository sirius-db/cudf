/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

namespace cudf::detail {

namespace {

/**
 * @brief Mean bytes-per-copy at or above which a batch asks the driver for raw copy
 * bandwidth instead of overlap with concurrent compute.
 *
 * `cudaMemcpyFlagPreferOverlapWithCompute` asks the driver to service the batch in a way that
 * leaves the device free for concurrent kernels, at the cost of copy bandwidth. That cost is
 * severe for large device-to-device copies. Measured D2D on GB300 / CUDA 13.2, sweeping the
 * per-copy size (the ratios are stable across batch sizes of 1, 4, 16 and 64):
 *
 *     bytes/copy | PreferOverlap | Default   | Default/PreferOverlap
 *       64 KiB   |    204 GB/s   |   38 GB/s |  0.19x
 *      256 KiB   |    431 GB/s   |  155 GB/s |  0.36x
 *      512 KiB   |    454 GB/s   |  311 GB/s |  0.69x
 *        1 MiB   |    462 GB/s   |  595 GB/s |  1.37x
 *        2 MiB   |    455 GB/s   | 1183 GB/s |  2.60x
 *       16 MiB   |    470 GB/s   | 2275 GB/s |  4.84x
 *      256 MiB   |    472 GB/s   | 3125 GB/s |  6.62x
 *
 * `PreferOverlap` saturates at ~470 GB/s no matter how large the copy is, while `Default`
 * reaches ~3.2 TB/s. Below ~512 KiB per copy the ordering reverses -- the driver fuses a batch
 * of small copies far more effectively under `PreferOverlap` (0.19x at 64 KiB) -- so the flag
 * must be kept for small copies.
 *
 * The crossover sits at ~1 MiB per copy and barely moves with the number of copies in the
 * batch, so the threshold is applied to the mean size rather than to the batch total: a batch
 * of many small buffers must keep the flag even when its total is large. 2 MiB is used rather
 * than the measured 1 MiB crossover to keep a margin over the near-tie there (1.03x for a
 * single 1 MiB copy).
 *
 * Host-device copies are insensitive to this flag -- pinned and pageable H2D were within 1% at
 * every size measured -- so in practice this only changes device-to-device behavior.
 */
constexpr std::size_t prefer_bandwidth_bytes_per_copy = 2ul * 1024 * 1024;

// Simple kernel to copy between device buffers
CUDF_KERNEL void copy_kernel(char const* __restrict__ src, char* __restrict__ dst, size_t n)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx < n) { dst[idx] = src[idx]; }
}

void copy_pinned(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  if (size < get_kernel_pinned_copy_threshold()) {
    const int block_size = 256;
    auto const grid_size = cudf::util::div_rounding_up_safe<size_t>(size, block_size);
    // We are explicitly launching the kernel here instead of calling a thrust function because the
    // thrust function can potentially call cudaMemcpyAsync instead of using a kernel
    copy_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      static_cast<char const*>(src), static_cast<char*>(dst), size);
  } else {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, stream));
  }
}

void copy_pageable(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, src, size, stream));
}

};  // namespace

cudaError_t memcpy_batch_async(void* const* dsts,
                               void const* const* srcs,
                               std::size_t const* sizes,
                               std::size_t count,
                               rmm::cuda_stream_view stream)
{
// Uses cudaMemcpyBatchAsync for CUDA 13.0+ to avoid driver-side locking overhead.
// cudaMemcpyBatchAsync does not support the default stream.
#if CUDART_VERSION >= 13000
  if (!stream.is_default()) {
    // Filter out invalid copies (nullptr dst/src or size==0);
    // cudaMemcpyBatchAsync does not support these inputs
    auto is_invalid = [&](auto i) {
      return dsts[i] == nullptr || srcs[i] == nullptr || sizes[i] == 0;
    };
    std::vector<void*> valid_dsts;
    std::vector<void const*> valid_srcs;
    std::vector<std::size_t> valid_sizes;

    if (std::ranges::any_of(std::ranges::views::iota(std::size_t{0}, count), is_invalid)) {
      valid_dsts.reserve(count);
      valid_srcs.reserve(count);
      valid_sizes.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        if (dsts[i] != nullptr && srcs[i] != nullptr && sizes[i] != 0) {
          valid_dsts.push_back(dsts[i]);
          valid_srcs.push_back(srcs[i]);
          valid_sizes.push_back(sizes[i]);
        }
      }
      if (valid_dsts.empty()) { return cudaSuccess; }
      dsts  = valid_dsts.data();
      srcs  = valid_srcs.data();
      sizes = valid_sizes.data();
      count = valid_dsts.size();
    }
    if (count == 0) { return cudaSuccess; }

    // Large copies are bandwidth-bound and are almost always on the critical path (e.g. the
    // buffer copies behind `cudf::concatenate`); small ones benefit from the driver's fused,
    // compute-friendly path. See `prefer_bandwidth_bytes_per_copy`.
    auto const total_bytes = std::accumulate(sizes, sizes + count, std::size_t{0});
    auto const prefer_bandwidth =
      (total_bytes / count) >= prefer_bandwidth_bytes_per_copy;

    cudaMemcpyAttributes attrs = {
      .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
      .flags          = static_cast<unsigned int>(prefer_bandwidth
                                                    ? cudaMemcpyFlagDefault
                                                    : cudaMemcpyFlagPreferOverlapWithCompute)};
    std::size_t attrs_idxs     = 0;
    return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attrs, &attrs_idxs, 1, stream.value());
  }
#endif  // CUDART_VERSION >= 13000
  for (std::size_t i = 0; i < count; ++i) {
    cudaError_t status =
      cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value());
    if (status != cudaSuccess) { return status; }
  }
  return cudaSuccess;
}

cudaError_t memcpy_async(void* dst, void const* src, size_t count, rmm::cuda_stream_view stream)
{
  if (count == 0) { return cudaSuccess; }

  // Use batch API with size 1 to prefer cudaMemcpyBatchAsync over
  // cudaMemcpyAsync. The batched API is more efficient.
  return memcpy_batch_async(&dst, &src, &count, 1, stream);
}

void cuda_memcpy_async_impl(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream)
{
  if (kind == host_memory_kind::PINNED) {
    copy_pinned(dst, src, size, stream);
  } else if (kind == host_memory_kind::PAGEABLE) {
    copy_pageable(dst, src, size, stream);
  } else {
    CUDF_FAIL("Unsupported host memory kind");
  }
}

}  // namespace cudf::detail
