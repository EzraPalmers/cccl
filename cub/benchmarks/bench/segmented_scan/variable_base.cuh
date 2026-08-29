// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <nvbench_helper.cuh>

namespace
{
inline constexpr cuda::std::uint64_t seed = 0xCCC1;

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr cuda::std::uint64_t
counter_u64(cuda::std::uint64_t index, cuda::std::uint64_t stream = 0) noexcept
{
  auto value = seed ^ (index * 0x9E3779B97F4A7C15ULL) ^ (stream * 0xD1B54A32D192ED03ULL);
  value += 0x9E3779B97F4A7C15ULL;
  value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL);
  value = ((value ^ (value >> 27)) * 0x94D049BB133111EBULL);
  return value ^ (value >> 31);
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double
counter_uniform(cuda::std::uint64_t index, cuda::std::uint64_t stream = 0) noexcept
{
  return static_cast<double>((counter_u64(index, stream) >> 11) + 0.5) / 9007199254740992.0;
}

struct even_weight
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double operator()(cuda::std::uint64_t) const noexcept
  {
    return 1.0;
  }
};

struct lognormal_weight
{
  double sigma;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(cuda::std::uint64_t index) const noexcept
  {
    const auto first  = counter_uniform(index, 0);
    const auto second = counter_uniform(index, 1);
    const auto normal =
      cuda::std::sqrt(-2.0 * cuda::std::log(first)) * cuda::std::cos(6.283185307179586476925286766559 * second);
    return cuda::std::exp(sigma * normal);
  }
};

struct pareto_weight
{
  double alpha;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(cuda::std::uint64_t index) const noexcept
  {
    return cuda::std::pow(1.0 - counter_uniform(index, 2), -1.0 / alpha);
  }
};

struct zipf_mass
{
  double exponent;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(cuda::std::uint64_t index) const noexcept
  {
    return cuda::std::pow(static_cast<double>(index + 1), -exponent);
  }
};

struct zipf_sample
{
  double total;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(cuda::std::uint64_t index) const noexcept
  {
    return counter_uniform(index, 3) * total;
  }
};

struct rank_to_weight
{
  template <typename OffsetT>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double operator()(OffsetT rank) const noexcept
  {
    return static_cast<double>(rank + 1);
  }
};

struct mixture_hash
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr cuda::std::uint64_t
  operator()(cuda::std::uint64_t index) const noexcept
  {
    return counter_u64(index, 4);
  }
};

template <typename OffsetT>
struct set_weight
{
  double* weights;
  double value;

  _CCCL_DEVICE_API void operator()(OffsetT index) const noexcept
  {
    weights[index] = value;
  }
};

struct valid_weight
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(double weight) const noexcept
  {
    return cuda::std::isfinite(weight) && weight > 0.0;
  }
};

struct fractional_length
{
  cuda::std::int64_t variable_budget;
  double inverse_weight_sum;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(double weight) const noexcept
  {
    const auto scaled = static_cast<double>(variable_budget) * weight * inverse_weight_sum;
    return scaled - cuda::std::floor(scaled);
  }
};

template <typename OffsetT>
struct integral_length
{
  cuda::std::int64_t variable_budget;
  double inverse_weight_sum;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OffsetT operator()(double weight) const noexcept
  {
    const auto scaled = static_cast<double>(variable_budget) * weight * inverse_weight_sum;
    return static_cast<OffsetT>(cuda::std::floor(scaled)) + OffsetT{1};
  }
};

template <typename OffsetT>
struct increment_length
{
  OffsetT* lengths;

  _CCCL_DEVICE_API void operator()(OffsetT index) const noexcept
  {
    ++lengths[index];
  }
};

template <typename OffsetT, typename Weight>
[[nodiscard]] thrust::device_vector<double> generate_weights(OffsetT num_segments, Weight weight)
{
  auto weights = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::tabulate(weights.begin(), weights.end(), weight);
  return weights;
}

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<double> generate_zipf_weights(OffsetT num_segments, double exponent)
{
  auto cumulative = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::tabulate(cumulative.begin(), cumulative.end(), zipf_mass{exponent});
  thrust::inclusive_scan(cumulative.begin(), cumulative.end(), cumulative.begin());
  const auto total = static_cast<double>(cumulative.back());

  auto samples = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::tabulate(samples.begin(), samples.end(), zipf_sample{total});

  auto ranks = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::lower_bound(cumulative.begin(), cumulative.end(), samples.begin(), samples.end(), ranks.begin());

  auto weights = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::transform(ranks.begin(), ranks.end(), weights.begin(), rank_to_weight{});
  return weights;
}

template <typename OffsetT>
[[nodiscard]] OffsetT rounded_mixture_count(OffsetT num_segments, double long_fraction)
{
  const auto exact      = static_cast<double>(num_segments) * long_fraction;
  const auto lower      = cuda::std::floor(exact);
  const auto fraction   = exact - lower;
  const auto lower_size = static_cast<OffsetT>(lower);
  const auto round_up   = fraction > 0.5 || (fraction == 0.5 && (lower_size & 1) != 0);
  return cuda::std::min(num_segments - 1, cuda::std::max(OffsetT{1}, lower_size + static_cast<OffsetT>(round_up)));
}

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<double>
generate_two_mode_weights(OffsetT num_segments, double long_fraction, double ratio)
{
  const auto long_count = rounded_mixture_count(num_segments, long_fraction);
  auto hashes           = thrust::device_vector<cuda::std::uint64_t>(num_segments, thrust::no_init);
  thrust::tabulate(hashes.begin(), hashes.end(), mixture_hash{});

  auto indices = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::sequence(indices.begin(), indices.end());
  thrust::stable_sort_by_key(hashes.begin(), hashes.end(), indices.begin());

  auto weights = thrust::device_vector<double>(num_segments, 1.0);
  thrust::for_each(
    indices.begin(),
    indices.begin() + long_count,
    set_weight<OffsetT>{thrust::raw_pointer_cast(weights.data()), ratio});
  return weights;
}

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT>
weights_to_offsets(OffsetT elements, OffsetT num_segments, const thrust::device_vector<double>& weights)
{
  if (weights.size() != static_cast<std::size_t>(num_segments)
      || !thrust::all_of(weights.begin(), weights.end(), valid_weight{}))
  {
    throw std::runtime_error("invalid variable segment weights");
  }

  const auto variable_budget = static_cast<cuda::std::int64_t>(elements) - num_segments;
  const auto weight_sum      = thrust::reduce(weights.begin(), weights.end(), 0.0);
  if (variable_budget < 0 || !cuda::std::isfinite(weight_sum) || weight_sum <= 0.0)
  {
    throw std::runtime_error("invalid variable segment budget");
  }

  const auto inverse_sum = 1.0 / weight_sum;
  auto lengths           = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::transform(
    weights.begin(), weights.end(), lengths.begin(), integral_length<OffsetT>{variable_budget, inverse_sum});

  const auto assigned  = thrust::reduce(lengths.begin(), lengths.end(), cuda::std::int64_t{0});
  const auto remainder = static_cast<OffsetT>(static_cast<cuda::std::int64_t>(elements) - assigned);
  if (remainder < 0 || remainder >= num_segments)
  {
    throw std::runtime_error("largest-remainder allocation is out of range");
  }

  auto fractions = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::transform(
    weights.begin(), weights.end(), fractions.begin(), fractional_length{variable_budget, inverse_sum});
  auto indices = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::sequence(indices.begin(), indices.end());
  thrust::stable_sort_by_key(fractions.begin(), fractions.end(), indices.begin(), thrust::greater<double>{});
  thrust::for_each(
    indices.begin(), indices.begin() + remainder, increment_length<OffsetT>{thrust::raw_pointer_cast(lengths.data())});

  const auto minimum = thrust::reduce(
    lengths.begin(), lengths.end(), cuda::std::numeric_limits<OffsetT>::max(), thrust::minimum<OffsetT>{});
  const auto total   = thrust::reduce(lengths.begin(), lengths.end(), cuda::std::int64_t{0});
  if (minimum < 1 || total != elements)
  {
    throw std::runtime_error("generated segment lengths violate their invariants");
  }

  auto offsets = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::fill_n(offsets.begin(), 1, OffsetT{0});
  thrust::inclusive_scan(lengths.begin(), lengths.end(), offsets.begin() + 1);
  return offsets;
}

struct two_mode_parameters
{
  double long_fraction;
  double ratio;
};

[[nodiscard]] two_mode_parameters string_to_two_mode(const std::string& regime)
{
  if (regime == "moderate_bimodal")
  {
    return {0.25, 10.0};
  }
  if (regime == "ten_percent_long")
  {
    return {0.10, 50.0};
  }
  if (regime == "few_huge")
  {
    return {0.02, 500.0};
  }
  throw std::runtime_error("Invalid Regime axis value: " + regime);
}

template <typename T, typename OffsetT>
void variable_segmented_scan(
  nvbench::state& state, nvbench::type_list<T, OffsetT>, const thrust::device_vector<double>& weights)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  const thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::default_init);
  const auto offsets = weights_to_offsets(elements, num_segments, weights);

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
    _CCCL_TRY_CUDA_API(
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

template <typename T, typename OffsetT>
void even_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto spread            = state.get_float64("Spread{io}");
  if (spread != 0.0)
  {
    throw std::runtime_error("even segment spread must be zero");
  }
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_weights(num_segments, even_weight{});
  variable_segmented_scan(state, tl, weights);
}

template <typename T, typename OffsetT>
void lognormal_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto sigma             = state.get_float64("Sigma{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_weights(num_segments, lognormal_weight{sigma});
  variable_segmented_scan(state, tl, weights);
}

template <typename T, typename OffsetT>
void pareto_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto alpha             = state.get_float64("Alpha{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_weights(num_segments, pareto_weight{alpha});
  variable_segmented_scan(state, tl, weights);
}

template <typename T, typename OffsetT>
void zipf_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto exponent          = state.get_float64("Exponent{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_zipf_weights(num_segments, exponent);
  variable_segmented_scan(state, tl, weights);
}

template <typename T, typename OffsetT>
void two_mode_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("SegmentSize{io}"));
  const auto parameters        = string_to_two_mode(state.get_string("Regime{io}"));
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights = generate_two_mode_weights(num_segments, parameters.long_fraction, parameters.ratio);
  variable_segmented_scan(state, tl, weights);
}
} // namespace

#ifdef TUNE_T
using variable_value_types = nvbench::type_list<TUNE_T>;
#else
using variable_value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

#ifdef TUNE_OffsetT
using variable_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using variable_offset_types = nvbench::type_list<int32_t>;
#endif

NVBENCH_BENCH_TYPES(even_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_even")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {16, 51, 123, 233, 513, 1337})
  .add_float64_axis("Spread{io}", {0.0});

NVBENCH_BENCH_TYPES(lognormal_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_lognormal")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {16, 51, 123, 233, 513, 1337})
  .add_float64_axis("Sigma{io}", {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5});

NVBENCH_BENCH_TYPES(pareto_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_pareto")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {16, 51, 123, 233, 513, 1337})
  .add_float64_axis("Alpha{io}", {5.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5});

NVBENCH_BENCH_TYPES(zipf_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_zipf")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {16, 51, 123, 233, 513, 1337})
  .add_float64_axis("Exponent{io}", {0.75, 1.0, 1.25, 1.5, 1.6, 2.0});

NVBENCH_BENCH_TYPES(two_mode_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_two_mode")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {16, 51, 123, 233, 513, 1337})
  .add_string_axis("Regime{io}", {"moderate_bimodal", "ten_percent_long", "few_huge"});
