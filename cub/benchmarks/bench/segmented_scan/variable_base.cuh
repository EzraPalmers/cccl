// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/random>

#include <cassert>

#include <nvbench_helper.cuh>

namespace segmented_scan_study
{
using offset_type = ::cuda::std::int32_t;
using value_type  = ::cuda::std::int32_t;
using seed_type   = ::cuda::std::philox4x32::result_type;

enum class generation_mode
{
  sampled,
  quantile
};

struct distribution_probability
{
  generation_mode mode;
  seed_type seed;
  ::cuda::std::uint64_t count;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    if (mode == generation_mode::quantile)
    {
      return (static_cast<double>(index) + 0.5) / static_cast<double>(count);
    }

    ::cuda::std::philox4x32 rng(seed);
    rng.set_counter({0, 0, static_cast<seed_type>(index), 0});
    ::cuda::std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(rng);
  }
};

struct lognormal_weight
{
  generation_mode mode;
  seed_type seed;
  ::cuda::std::uint64_t count;
  double sigma;

  [[nodiscard]] _CCCL_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    if (sigma == 0.0)
    {
      return 1.0;
    }
    if (mode == generation_mode::quantile)
    {
      const auto probability = (static_cast<double>(index) + 0.5) / static_cast<double>(count);
      return ::cuda::std::exp(sigma * ::normcdfinv(probability));
    }

    ::cuda::std::philox4x32 rng(seed);
    rng.set_counter({0, 0, static_cast<seed_type>(index), 0});
    return ::cuda::std::exp(::cuda::std::normal_distribution<double>{0.0, sigma}(rng));
  }
};

struct pareto_weight
{
  distribution_probability probability;
  double alpha;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    return ::cuda::std::pow(1.0 - probability(index), -1.0 / alpha);
  }
};

struct zipf_mass
{
  double exponent;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    return ::cuda::std::pow(static_cast<double>(index + 1), -exponent);
  }
};

struct probability_scale
{
  distribution_probability probability;
  double total;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    return probability(index) * total;
  }
};

struct rank_to_weight
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(offset_type rank) const noexcept
  {
    return static_cast<double>(rank + 1);
  }
};

// Sampled mode compares a per-index draw against LongSegmentFraction; quantile mode compares the
// deterministic quantile position instead, so both share this one functor.
struct multimodal_weight
{
  distribution_probability probability;
  double long_fraction;
  double long_to_short_ratio;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(::cuda::std::uint64_t index) const noexcept
  {
    return probability(index) < long_fraction ? long_to_short_ratio : 1.0;
  }
};

// Quantile mode uses this rank threshold directly rather than routing through
// distribution_probability's quantile branch, avoiding a division per index.
struct deterministic_multimodal_weight
{
  offset_type long_count;
  double long_to_short_ratio;

  [[nodiscard]] _CCCL_HOST_DEVICE_API double operator()(offset_type index) const noexcept
  {
    return index < long_count ? long_to_short_ratio : 1.0;
  }
};

template <typename Weight>
[[nodiscard]] thrust::device_vector<double> make_weights(offset_type count, Weight weight)
{
  auto weights = thrust::device_vector<double>(count, thrust::no_init);
  thrust::tabulate(weights.begin(), weights.end(), weight);
  return weights;
}

// Keys the shuffle by ShuffleSeed and streams it by GenerationSeed, so no (GenerationSeed,
// ShuffleSeed) pair collides with another. Both vary as real axes on this branch.
inline void shuffle_weights(thrust::device_vector<double>& weights, seed_type shuffle_seed, seed_type generation_seed)
{
  ::cuda::std::philox4x32 rng(shuffle_seed);
  rng.set_counter({0, 0, generation_seed, 0});
  thrust::shuffle(weights.begin(), weights.end(), rng);
}

[[nodiscard]] inline thrust::device_vector<double> make_lognormal_weights(
  offset_type count, double sigma, seed_type generation_seed, seed_type shuffle_seed, generation_mode mode)
{
  auto weights =
    make_weights(count, lognormal_weight{mode, generation_seed, static_cast<::cuda::std::uint64_t>(count), sigma});
  shuffle_weights(weights, shuffle_seed, generation_seed);
  return weights;
}

[[nodiscard]] inline thrust::device_vector<double> make_pareto_weights(
  offset_type count, double alpha, seed_type generation_seed, seed_type shuffle_seed, generation_mode mode)
{
  auto weights =
    make_weights(count, pareto_weight{{mode, generation_seed, static_cast<::cuda::std::uint64_t>(count)}, alpha});
  shuffle_weights(weights, shuffle_seed, generation_seed);
  return weights;
}

[[nodiscard]] inline thrust::device_vector<double> make_zipf_weights(
  offset_type count, double exponent, seed_type generation_seed, seed_type shuffle_seed, generation_mode mode)
{
  auto cumulative = make_weights(count, zipf_mass{exponent});
  thrust::inclusive_scan(cumulative.begin(), cumulative.end(), cumulative.begin());
  const auto total = static_cast<double>(cumulative.back());

  auto samples =
    make_weights(count, probability_scale{{mode, generation_seed, static_cast<::cuda::std::uint64_t>(count)}, total});
  auto ranks = thrust::device_vector<offset_type>(count, thrust::no_init);
  thrust::lower_bound(cumulative.begin(), cumulative.end(), samples.begin(), samples.end(), ranks.begin());

  auto weights = thrust::device_vector<double>(count, thrust::no_init);
  thrust::transform(ranks.begin(), ranks.end(), weights.begin(), rank_to_weight{});
  shuffle_weights(weights, shuffle_seed, generation_seed);
  return weights;
}

// No correction is applied for the fixed weight ratio overshooting the requested
// LongSegmentFraction/LongToShortRatio at small mean segment sizes; the realised long-to-short
// ratio drifts from the requested one there. testing's compensated_multimodal_weight_ratio is not
// ported.
[[nodiscard]] inline thrust::device_vector<double> make_multimodal_weights(
  offset_type count,
  double long_fraction,
  double long_to_short_ratio,
  seed_type generation_seed,
  seed_type shuffle_seed,
  generation_mode mode)
{
  thrust::device_vector<double> weights;
  if (mode == generation_mode::sampled)
  {
    weights = make_weights(
      count,
      multimodal_weight{
        {mode, generation_seed, static_cast<::cuda::std::uint64_t>(count)}, long_fraction, long_to_short_ratio});
  }
  else
  {
    const auto long_count = static_cast<offset_type>(::cuda::std::round(long_fraction * static_cast<double>(count)));
    weights               = make_weights(count, deterministic_multimodal_weight{long_count, long_to_short_ratio});
  }
  shuffle_weights(weights, shuffle_seed, generation_seed);
  return weights;
}

struct cumulative_to_offset
{
  const double* cumulative_weights;
  double inverse_weight_sum;
  ::cuda::std::int64_t variable_budget;
  offset_type elements;
  offset_type count;

  [[nodiscard]] _CCCL_HOST_DEVICE_API offset_type operator()(offset_type index) const noexcept
  {
    if (index == 0)
    {
      return 0;
    }
    if (index == count)
    {
      return elements;
    }

    const auto variable_offset =
      ::cuda::std::round(static_cast<double>(variable_budget) * cumulative_weights[index] * inverse_weight_sum);
    return index + static_cast<offset_type>(variable_offset);
  }
};

[[nodiscard]] inline thrust::device_vector<offset_type> weights_to_offsets(
  nvbench::state& state, offset_type elements, offset_type count, const thrust::device_vector<double>& weights)
{
  const auto minimum_total = static_cast<::cuda::std::int64_t>(count);
  if (static_cast<::cuda::std::int64_t>(elements) < minimum_total)
  {
    state.skip("element count is smaller than the minimum segment allocation");
    return {};
  }

  const auto weight_sum = thrust::reduce(weights.begin(), weights.end(), 0.0);
  assert(weight_sum > 0.0);

  auto cumulative = thrust::device_vector<double>(count + 1, thrust::no_init);
  thrust::exclusive_scan(weights.begin(), weights.end(), cumulative.begin());
  thrust::fill_n(cumulative.end() - 1, 1, weight_sum);

  const auto variable_budget = static_cast<::cuda::std::int64_t>(elements) - minimum_total;
  auto offsets               = thrust::device_vector<offset_type>(count + 1, thrust::no_init);
  thrust::tabulate(offsets.begin(),
                   offsets.end(),
                   cumulative_to_offset{
                     thrust::raw_pointer_cast(cumulative.data()), 1.0 / weight_sum, variable_budget, elements, count});
  return offsets;
}

inline void run(nvbench::state& state, const thrust::device_vector<double>& weights)
{
  const auto elements          = static_cast<offset_type>(state.get_int64("Elements{io}"));
  const auto mean_segment_size = static_cast<offset_type>(state.get_int64("MeanSegmentSize{io}"));
  const auto count             = ::cuda::ceil_div(elements, mean_segment_size);

  auto& summary = state.add_summary("user/derived/segment_count");
  summary.set_string("name", "#Segments");
  summary.set_int64("value", count);

  const thrust::device_vector<value_type> input = generate(elements);
  thrust::device_vector<value_type> output(elements, thrust::default_init);
  const auto offsets = weights_to_offsets(state, elements, count, weights);
  if (offsets.empty())
  {
    return;
  }

  const value_type* d_input  = thrust::raw_pointer_cast(input.data());
  value_type* d_output       = thrust::raw_pointer_cast(output.data());
  const offset_type* d_begin = thrust::raw_pointer_cast(offsets.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<value_type>(elements);
  state.add_global_memory_reads<offset_type>(count + 1);
  state.add_global_memory_writes<value_type>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSegmentedScan::ExclusiveSegmentedScan,
      "ExclusiveSegmentedScan failed",
      d_input,
      d_output,
      d_begin,
      d_begin + 1,
      d_begin,
      count,
      ::cuda::std::plus<>{},
      value_type{},
      env);
  });
}

[[nodiscard]] inline offset_type segment_count(nvbench::state& state)
{
  const auto elements = static_cast<offset_type>(state.get_int64("Elements{io}"));
  const auto mean     = static_cast<offset_type>(state.get_int64("MeanSegmentSize{io}"));
  return ::cuda::ceil_div(elements, mean);
}

[[nodiscard]] inline seed_type study_generation_seed(nvbench::state& state)
{
  return static_cast<seed_type>(state.get_int64("GenerationSeed{io}"));
}

[[nodiscard]] inline seed_type study_shuffle_seed(nvbench::state& state)
{
  return static_cast<seed_type>(state.get_int64("ShuffleSeed{io}"));
}

[[nodiscard]] inline bool is_study_cell(nvbench::state& state)
{
  const auto elements = state.get_int64("Elements{io}");
  const auto mean     = state.get_int64("MeanSegmentSize{io}");
  if ((elements == (1LL << 22) && (mean == 128 || mean == 256 || mean == 512))
      || (elements == (1LL << 26) && (mean == 256 || mean == 512 || mean == 2048)))
  {
    return true;
  }
  state.skip("element count and mean segment size are not a study cell");
  return false;
}
} // namespace segmented_scan_study
