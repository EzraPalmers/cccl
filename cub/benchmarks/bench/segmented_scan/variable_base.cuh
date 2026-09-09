// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/random>

#include <cassert>

#include <nvbench_helper.cuh>

namespace
{
struct normal_sample
{
  double sigma;
  ::cuda::std::philox4x32::result_type seed;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    if (sigma == 0.0)
    {
      return 0.0;
    }

    ::cuda::std::philox4x32 rng(seed);
    rng.set_counter({0, 0, static_cast<::cuda::std::philox4x32::result_type>(index), 0});
    ::cuda::std::normal_distribution<double> normal(0.0, sigma);
    return normal(rng);
  }
};

struct lognormal_weight
{
  double maximum;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(double sample) const noexcept
  {
    return ::cuda::std::exp(sample - maximum);
  }
};

template <typename OffsetT>
struct cumulative_to_offset
{
  const double* cumulative_weights;
  double inverse_weight_sum;
  ::cuda::std::int64_t variable_budget;
  OffsetT elements;
  OffsetT num_segments;
  OffsetT minimum_segment_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OffsetT operator()(OffsetT index) const noexcept
  {
    if (index == 0)
    {
      return 0;
    }
    if (index == num_segments)
    {
      return elements;
    }

    const auto variable_offset =
      ::cuda::std::round(static_cast<double>(variable_budget) * cumulative_weights[index] * inverse_weight_sum);
    return index * minimum_segment_size + static_cast<OffsetT>(variable_offset);
  }
};

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT> generate_lognormal_segment_offsets(
  nvbench::state& state, OffsetT elements, OffsetT num_segments, OffsetT minimum_segment_size, double sigma)
{
  const auto minimum_total =
    static_cast<::cuda::std::int64_t>(num_segments) * static_cast<::cuda::std::int64_t>(minimum_segment_size);
  if (static_cast<::cuda::std::int64_t>(elements) < minimum_total)
  {
    state.skip("element count is smaller than the minimum segment allocation");
    return {};
  }

  auto samples = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::tabulate(samples.begin(),
                   samples.end(),
                   normal_sample{sigma, static_cast<::cuda::std::philox4x32::result_type>(seed_t{}.get())});

  const auto maximum = thrust::reduce(
    samples.begin(), samples.end(), ::cuda::std::numeric_limits<double>::lowest(), ::cuda::maximum<double>{});
  auto weights = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::transform(samples.begin(), samples.end(), weights.begin(), lognormal_weight{maximum});

  const auto weight_sum = thrust::reduce(weights.begin(), weights.end(), 0.0);
  assert(weight_sum > 0.0);

  auto cumulative_weights = thrust::device_vector<double>(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(weights.begin(), weights.end(), cumulative_weights.begin());
  thrust::fill_n(cumulative_weights.end() - 1, 1, weight_sum);

  const auto variable_budget = static_cast<::cuda::std::int64_t>(elements) - minimum_total;
  auto offsets               = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::tabulate(
    offsets.begin(),
    offsets.end(),
    cumulative_to_offset<OffsetT>{
      thrust::raw_pointer_cast(cumulative_weights.data()),
      1.0 / weight_sum,
      variable_budget,
      elements,
      num_segments,
      minimum_segment_size});

  assert(thrust::is_sorted(offsets.begin(), offsets.end()));
  return offsets;
}

template <typename T, typename OffsetT>
void skewed_size_segments(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto sigma             = state.get_float64("Sigma{io}");
  const auto num_segments      = ::cuda::ceil_div(elements, mean_segment_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  const thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::default_init);
  const auto offsets = generate_lognormal_segment_offsets(state, elements, num_segments, OffsetT{1}, sigma);
  if (offsets.empty())
  {
    return;
  }

  const T* d_input         = thrust::raw_pointer_cast(input.data());
  T* d_output              = thrust::raw_pointer_cast(output.data());
  const OffsetT* d_offsets = thrust::raw_pointer_cast(offsets.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<OffsetT>(num_segments + 1);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSegmentedScan::ExclusiveSegmentedScan,
      "ExclusiveSegmentedScan failed",
      d_input,
      d_output,
      d_offsets,
      d_offsets + 1,
      d_offsets,
      num_segments,
      op_t{},
      T{},
      env);
  });
}
} // namespace

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t>;
#endif

NVBENCH_BENCH_TYPES(skewed_size_segments, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("skewed_size_segments_small")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512})
  .add_float64_axis("Sigma{io}", {0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0});

NVBENCH_BENCH_TYPES(skewed_size_segments, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("skewed_size_segments_large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Sigma{io}", {0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0});
