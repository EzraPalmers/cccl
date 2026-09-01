// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/functional>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <cstddef>
#include <string>

#include <nvbench_helper.cuh>

namespace
{
struct default_variable_benchmark_traits
{
  using value_types = nvbench::type_list<int32_t, int64_t, float, double>;

  template <typename T, typename OffsetT>
  [[nodiscard]] static thrust::device_vector<T> make_input(OffsetT elements)
  {
    return generate(elements);
  }

  template <typename T, typename OffsetT>
  static void validate(const thrust::device_vector<T>&,
                       const thrust::device_vector<T>&,
                       const thrust::device_vector<OffsetT>&) noexcept
  {}
};

#ifdef VARIABLE_BENCHMARK_TRAITS
using variable_benchmark_traits = VARIABLE_BENCHMARK_TRAITS;
#else
using variable_benchmark_traits = default_variable_benchmark_traits;
#endif

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

struct multimodal_hash
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr cuda::std::uint64_t
  operator()(cuda::std::uint64_t index) const noexcept
  {
    return counter_u64(index, 4);
  }
};

struct shuffled_hash
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr cuda::std::uint64_t
  operator()(cuda::std::uint64_t index) const noexcept
  {
    return counter_u64(index, 5);
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
[[nodiscard]] OffsetT rounded_multimodal_count(OffsetT num_segments, double long_fraction)
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
generate_multimodal_weights(OffsetT num_segments, OffsetT long_count, double weight_ratio)
{
  auto hashes = thrust::device_vector<cuda::std::uint64_t>(num_segments, thrust::no_init);
  thrust::tabulate(hashes.begin(), hashes.end(), multimodal_hash{});

  auto indices = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::sequence(indices.begin(), indices.end());
  thrust::stable_sort_by_key(hashes.begin(), hashes.end(), indices.begin());

  auto weights = thrust::device_vector<double>(num_segments, 1.0);
  thrust::for_each(
    indices.begin(),
    indices.begin() + long_count,
    set_weight<OffsetT>{thrust::raw_pointer_cast(weights.data()), weight_ratio});
  return weights;
}

template <typename OffsetT>
[[nodiscard]] bool apply_segment_ordering(
  nvbench::state& state,
  thrust::device_vector<OffsetT>& lengths,
  const std::string& ordering,
  const OffsetT long_count)
{
  if (ordering == "as_sampled")
  {
    return true;
  }
  if (ordering == "ascending")
  {
    thrust::stable_sort(lengths.begin(), lengths.end());
    return true;
  }
  if (ordering == "descending")
  {
    thrust::stable_sort(lengths.begin(), lengths.end(), ::cuda::std::greater<OffsetT>{});
    return true;
  }
  if (ordering == "shuffled")
  {
    auto hashes = thrust::device_vector<cuda::std::uint64_t>(lengths.size(), thrust::no_init);
    thrust::tabulate(hashes.begin(), hashes.end(), shuffled_hash{});
    thrust::stable_sort_by_key(hashes.begin(), hashes.end(), lengths.begin());
    return true;
  }
  if (ordering == "clustered")
  {
    auto sorted_lengths = lengths;
    thrust::stable_sort(sorted_lengths.begin(), sorted_lengths.end(), ::cuda::std::greater<OffsetT>{});

    auto hashes = thrust::device_vector<cuda::std::uint64_t>(lengths.size(), thrust::no_init);
    thrust::tabulate(hashes.begin(), hashes.end(), shuffled_hash{});
    thrust::stable_sort_by_key(hashes.begin(), hashes.begin() + long_count, sorted_lengths.begin());
    thrust::stable_sort_by_key(
      hashes.begin() + long_count, hashes.end(), sorted_lengths.begin() + long_count);

    const auto num_segments   = static_cast<OffsetT>(lengths.size());
    const auto rest_count     = num_segments - long_count;
    const auto cluster_count  = cuda::std::max(OffsetT{1}, cuda::ceil_div(long_count, OffsetT{100}));
    const auto long_base      = long_count / cluster_count;
    const auto long_remainder = long_count % cluster_count;
    const auto rest_base       = rest_count / cluster_count;
    const auto rest_remainder  = rest_count % cluster_count;

    thrust::host_vector<OffsetT> host_indices;
    host_indices.reserve(lengths.size());
    OffsetT long_index = 0;
    OffsetT rest_index = long_count;
    for (OffsetT cluster = 0; cluster < cluster_count; ++cluster)
    {
      const auto long_run = long_base + static_cast<OffsetT>(cluster < long_remainder);
      for (OffsetT index = 0; index < long_run; ++index)
      {
        host_indices.push_back(long_index++);
      }

      const auto gap = rest_base + static_cast<OffsetT>(cluster < rest_remainder);
      for (OffsetT index = 0; index < gap; ++index)
      {
        host_indices.push_back(rest_index++);
      }
    }

    const thrust::device_vector<OffsetT> indices = host_indices;
    auto ordered_lengths = thrust::device_vector<OffsetT>(lengths.size(), thrust::no_init);
    thrust::gather(indices.begin(), indices.end(), sorted_lengths.begin(), ordered_lengths.begin());
    lengths.swap(ordered_lengths);
    return true;
  }

  state.skip("unknown segment ordering");
  return false;
}

template <typename OffsetT>
void add_realised_shape_summaries(nvbench::state& state, const thrust::device_vector<OffsetT>& lengths)
{
  const thrust::host_vector<OffsetT> host_lengths = lengths;
  cuda::std::int64_t total_length = 0;
  OffsetT max_segment_length      = 0;
  for (const auto segment_length : host_lengths)
  {
    total_length += segment_length;
    max_segment_length = ::cuda::std::max(max_segment_length, segment_length);
  }

  const auto mean = static_cast<double>(total_length) / static_cast<double>(host_lengths.size());
  double squared_deviation_sum = 0.0;
  for (const auto segment_length : host_lengths)
  {
    const auto deviation = static_cast<double>(segment_length) - mean;
    squared_deviation_sum += deviation * deviation;
  }

  const auto coefficient_of_variation =
    cuda::std::sqrt(squared_deviation_sum / static_cast<double>(host_lengths.size())) / mean;

  auto& cv_summary = state.add_summary("user/derived/realised_segment_length_cv");
  cv_summary.set_string("name", "RealisedSegmentLengthCV");
  cv_summary.set_float64("value", coefficient_of_variation);

  auto& max_summary = state.add_summary("user/derived/realised_max_segment_length");
  max_summary.set_string("name", "RealisedMaxSegmentLength");
  max_summary.set_int64("value", max_segment_length);
}

template <typename OffsetT>
void add_realised_ordering_summary(nvbench::state& state, const thrust::device_vector<OffsetT>& lengths)
{
  const thrust::host_vector<OffsetT> host_lengths = lengths;
  double correlation = 0.0;
  if (host_lengths.size() >= 2)
  {
    cuda::std::int64_t total_length = 0;
    for (const auto segment_length : host_lengths)
    {
      total_length += segment_length;
    }
    const auto mean = static_cast<double>(total_length) / static_cast<double>(host_lengths.size());

    double adjacent_product_sum       = 0.0;
    double leading_squared_sum        = 0.0;
    double trailing_squared_sum       = 0.0;
    for (std::size_t index = 0; index + 1 < host_lengths.size(); ++index)
    {
      const auto leading_deviation  = static_cast<double>(host_lengths[index]) - mean;
      const auto trailing_deviation = static_cast<double>(host_lengths[index + 1]) - mean;
      adjacent_product_sum += leading_deviation * trailing_deviation;
      leading_squared_sum += leading_deviation * leading_deviation;
      trailing_squared_sum += trailing_deviation * trailing_deviation;
    }

    const auto denominator = cuda::std::sqrt(leading_squared_sum * trailing_squared_sum);
    if (denominator != 0.0)
    {
      correlation = adjacent_product_sum / denominator;
    }
  }

  auto& summary = state.add_summary("user/derived/realised_segment_length_lag1_correlation");
  summary.set_string("name", "RealisedSegmentLengthLag1Correlation");
  summary.set_float64("value", correlation);
}

template <typename OffsetT>
[[nodiscard]] thrust::device_vector<OffsetT>
weights_to_offsets(
  nvbench::state& state,
  OffsetT elements,
  OffsetT num_segments,
  const thrust::device_vector<double>& weights,
  const std::string& ordering,
  const OffsetT long_count)
{
  if (weights.size() != static_cast<std::size_t>(num_segments)
      || !thrust::all_of(weights.begin(), weights.end(), valid_weight{}))
  {
    state.skip("invalid variable segment weights");
    return {};
  }

  const auto variable_budget = static_cast<cuda::std::int64_t>(elements) - num_segments;
  const auto weight_sum      = thrust::reduce(weights.begin(), weights.end(), 0.0);
  if (variable_budget < 0 || !cuda::std::isfinite(weight_sum) || weight_sum <= 0.0)
  {
    state.skip("invalid variable segment budget");
    return {};
  }

  const auto inverse_sum = 1.0 / weight_sum;
  auto lengths           = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::transform(
    weights.begin(), weights.end(), lengths.begin(), integral_length<OffsetT>{variable_budget, inverse_sum});

  const auto assigned  = thrust::reduce(lengths.begin(), lengths.end(), cuda::std::int64_t{0});
  const auto remainder = static_cast<OffsetT>(static_cast<cuda::std::int64_t>(elements) - assigned);
  if (remainder < 0 || remainder >= num_segments)
  {
    state.skip("largest-remainder allocation is out of range");
    return {};
  }

  auto fractions = thrust::device_vector<double>(num_segments, thrust::no_init);
  thrust::transform(
    weights.begin(), weights.end(), fractions.begin(), fractional_length{variable_budget, inverse_sum});
  auto indices = thrust::device_vector<OffsetT>(num_segments, thrust::no_init);
  thrust::sequence(indices.begin(), indices.end());
  thrust::stable_sort_by_key(fractions.begin(), fractions.end(), indices.begin(), ::cuda::std::greater<double>{});
  thrust::for_each(
    indices.begin(), indices.begin() + remainder, increment_length<OffsetT>{thrust::raw_pointer_cast(lengths.data())});

  const auto minimum = thrust::reduce(
    lengths.begin(), lengths.end(), cuda::std::numeric_limits<OffsetT>::max(), ::cuda::minimum<OffsetT>{});
  const auto total   = thrust::reduce(lengths.begin(), lengths.end(), cuda::std::int64_t{0});
  if (minimum < 1 || total != elements)
  {
    state.skip("generated segment lengths violate their invariants");
    return {};
  }

  add_realised_shape_summaries(state, lengths);
  if (!apply_segment_ordering(state, lengths, ordering, long_count))
  {
    return {};
  }
  add_realised_ordering_summary(state, lengths);

  auto offsets = thrust::device_vector<OffsetT>(num_segments + 1, thrust::no_init);
  thrust::fill_n(offsets.begin(), 1, OffsetT{0});
  thrust::inclusive_scan(lengths.begin(), lengths.end(), offsets.begin() + 1);
  if (static_cast<OffsetT>(offsets.back()) != elements)
  {
    state.skip("ordered segment offsets do not cover all elements");
    return {};
  }
  return offsets;
}

template <typename OffsetT>
[[nodiscard]] double compensated_multimodal_weight_ratio(
  OffsetT elements, OffsetT num_segments, OffsetT long_count, double requested_ratio) noexcept
{
  const auto variable_budget = static_cast<cuda::std::int64_t>(elements) - num_segments;
  const auto short_count     = num_segments - long_count;
  const auto denominator     = static_cast<double>(variable_budget) - (requested_ratio - 1.0) * long_count;

  if (denominator <= 0.0)
  {
    return 0.0;
  }

  const auto numerator =
    (requested_ratio - 1.0) * static_cast<double>(short_count)
    + requested_ratio * static_cast<double>(variable_budget);
  return numerator / denominator;
}

template <typename T, typename OffsetT>
void variable_segmented_scan(
  nvbench::state& state,
  nvbench::type_list<T, OffsetT>,
  const thrust::device_vector<double>& weights,
  const OffsetT long_count)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto ordering          = state.get_string("SegmentOrdering{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", num_segments);

  const thrust::device_vector<T> input = variable_benchmark_traits::make_input<T>(elements);
  thrust::device_vector<T> output(elements, thrust::default_init);
  const auto offsets = weights_to_offsets(state, elements, num_segments, weights, ordering, long_count);
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
  variable_benchmark_traits::validate(input, output, offsets);
}

template <typename T, typename OffsetT>
void lognormal_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto sigma             = state.get_float64("Sigma{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_weights(num_segments, lognormal_weight{sigma});
  variable_segmented_scan(state, tl, weights, cuda::ceil_div(num_segments, OffsetT{4}));
}

template <typename T, typename OffsetT>
void pareto_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto alpha             = state.get_float64("Alpha{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_weights(num_segments, pareto_weight{alpha});
  variable_segmented_scan(state, tl, weights, cuda::ceil_div(num_segments, OffsetT{4}));
}

template <typename T, typename OffsetT>
void zipf_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto exponent          = state.get_float64("Exponent{io}");
  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto weights           = generate_zipf_weights(num_segments, exponent);
  variable_segmented_scan(state, tl, weights, cuda::ceil_div(num_segments, OffsetT{4}));
}

template <typename T, typename OffsetT>
void multimodal_segments(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements          = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<OffsetT>(state.get_int64("MeanSegmentSize{io}"));
  const auto long_fraction     = state.get_float64("LongSegmentFraction{io}");
  const auto requested_ratio   = state.get_float64("LongToShortRatio{io}");

  constexpr double minimum_short_segment_size = 16.0;
  const auto minimum_mean_segment_size =
    minimum_short_segment_size * ((1.0 - long_fraction) + long_fraction * requested_ratio);
  if (static_cast<double>(mean_segment_size) < minimum_mean_segment_size)
  {
    state.skip("requested multimodal short segment size is below 16 elements");
    return;
  }

  const auto num_segments      = cuda::ceil_div(elements, mean_segment_size);
  const auto long_count        = rounded_multimodal_count(num_segments, long_fraction);
  const auto weight_ratio =
    compensated_multimodal_weight_ratio(elements, num_segments, long_count, requested_ratio);

  const auto weights = generate_multimodal_weights(num_segments, long_count, weight_ratio);
  variable_segmented_scan(state, tl, weights, long_count);
}
} // namespace

using variable_value_types  = variable_benchmark_traits::value_types;
using variable_offset_types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(lognormal_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_lognormal")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis(
    "MeanSegmentSize{io}", {32, 51, 64, 123, 128, 233, 256, 512, 513, 1024, 1337, 2048, 4096, 8192, 16384})
  .add_string_axis("SegmentOrdering{io}", {"as_sampled", "ascending", "descending", "shuffled", "clustered"})
  .add_float64_axis("Sigma{io}", {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5});

NVBENCH_BENCH_TYPES(pareto_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_pareto")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis(
    "MeanSegmentSize{io}", {32, 51, 64, 123, 128, 233, 256, 512, 513, 1024, 1337, 2048, 4096, 8192, 16384})
  .add_string_axis("SegmentOrdering{io}", {"as_sampled", "ascending", "descending", "shuffled", "clustered"})
  .add_float64_axis("Alpha{io}", {5.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5});

NVBENCH_BENCH_TYPES(zipf_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_zipf")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis(
    "MeanSegmentSize{io}", {32, 51, 64, 123, 128, 233, 256, 512, 513, 1024, 1337, 2048, 4096, 8192, 16384})
  .add_string_axis("SegmentOrdering{io}", {"as_sampled", "ascending", "descending", "shuffled", "clustered"})
  .add_float64_axis("Exponent{io}", {0.75, 1.0, 1.25, 1.5, 1.6, 2.0});

NVBENCH_BENCH_TYPES(multimodal_segments, NVBENCH_TYPE_AXES(variable_value_types, variable_offset_types))
  .set_name("ragged_multimodal")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis(
    "MeanSegmentSize{io}", {32, 51, 64, 123, 128, 233, 256, 512, 513, 1024, 1337, 2048, 4096, 8192, 16384})
  .add_string_axis("SegmentOrdering{io}", {"as_sampled", "ascending", "descending", "shuffled", "clustered"})
  .add_float64_axis("LongSegmentFraction{io}", {0.25, 0.10, 0.02})
  .add_float64_axis("LongToShortRatio{io}", {10.0, 50.0, 100.0});
