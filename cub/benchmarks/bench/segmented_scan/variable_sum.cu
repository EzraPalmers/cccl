// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "variable_base.cuh"

namespace
{
template <segmented_scan_study::generation_mode Mode>
void lognormal(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count        = segmented_scan_study::segment_count(state);
  const auto sigma        = state.get_float64("Sigma{io}");
  const auto shuffle_seed = segmented_scan_study::study_shuffle_seed(state);
  segmented_scan_study::seed_type generation_seed{0};
  if constexpr (Mode == segmented_scan_study::generation_mode::sampled)
  {
    generation_seed = segmented_scan_study::study_generation_seed(state);
  }
  const auto weights = segmented_scan_study::make_lognormal_weights(count, sigma, generation_seed, shuffle_seed, Mode);
  segmented_scan_study::run(state, weights);
}

void lognormal_sampled(nvbench::state& state)
{
  lognormal<segmented_scan_study::generation_mode::sampled>(state);
}

void lognormal_quantile(nvbench::state& state)
{
  lognormal<segmented_scan_study::generation_mode::quantile>(state);
}
} // namespace

NVBENCH_BENCH(lognormal_sampled)
  .set_name("lognormal_sampled")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Sigma{io}", {0.0, 1.0, 1.25, 1.5, 2.0, 2.5})
  .add_int64_axis("GenerationSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

NVBENCH_BENCH(lognormal_quantile)
  .set_name("lognormal_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Sigma{io}", {0.0, 1.0, 1.25, 1.5, 2.0, 2.5})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});

namespace
{
template <segmented_scan_study::generation_mode Mode>
void pareto(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count        = segmented_scan_study::segment_count(state);
  const auto alpha        = state.get_float64("Alpha{io}");
  const auto shuffle_seed = segmented_scan_study::study_shuffle_seed(state);
  segmented_scan_study::seed_type generation_seed{0};
  if constexpr (Mode == segmented_scan_study::generation_mode::sampled)
  {
    generation_seed = segmented_scan_study::study_generation_seed(state);
  }
  const auto weights = segmented_scan_study::make_pareto_weights(count, alpha, generation_seed, shuffle_seed, Mode);
  segmented_scan_study::run(state, weights);
}

void pareto_sampled(nvbench::state& state)
{
  pareto<segmented_scan_study::generation_mode::sampled>(state);
}

void pareto_quantile(nvbench::state& state)
{
  pareto<segmented_scan_study::generation_mode::quantile>(state);
}
} // namespace

NVBENCH_BENCH(pareto_sampled)
  .set_name("pareto_sampled")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Alpha{io}", {1.3, 1.5, 1.75, 2.0, 2.5, 3.0})
  .add_int64_axis("GenerationSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

NVBENCH_BENCH(pareto_quantile)
  .set_name("pareto_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Alpha{io}", {1.3, 1.5, 1.75, 2.0, 2.5, 3.0})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});

namespace
{
template <segmented_scan_study::generation_mode Mode>
void zipf(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count        = segmented_scan_study::segment_count(state);
  const auto exponent     = state.get_float64("Exponent{io}");
  const auto shuffle_seed = segmented_scan_study::study_shuffle_seed(state);
  segmented_scan_study::seed_type generation_seed{0};
  if constexpr (Mode == segmented_scan_study::generation_mode::sampled)
  {
    generation_seed = segmented_scan_study::study_generation_seed(state);
  }
  const auto weights = segmented_scan_study::make_zipf_weights(count, exponent, generation_seed, shuffle_seed, Mode);
  segmented_scan_study::run(state, weights);
}

void zipf_sampled(nvbench::state& state)
{
  zipf<segmented_scan_study::generation_mode::sampled>(state);
}

void zipf_quantile(nvbench::state& state)
{
  zipf<segmented_scan_study::generation_mode::quantile>(state);
}
} // namespace

NVBENCH_BENCH(zipf_sampled)
  .set_name("zipf_sampled")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.25, 1.5, 1.75, 1.9})
  .add_int64_axis("GenerationSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

NVBENCH_BENCH(zipf_quantile)
  .set_name("zipf_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.25, 1.5, 1.75, 1.9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});

namespace
{
template <segmented_scan_study::generation_mode Mode>
void multimodal(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count               = segmented_scan_study::segment_count(state);
  const auto long_fraction       = state.get_float64("LongSegmentFraction{io}");
  const auto long_to_short_ratio = state.get_float64("LongToShortRatio{io}");
  const auto shuffle_seed        = segmented_scan_study::study_shuffle_seed(state);
  segmented_scan_study::seed_type generation_seed{0};
  if constexpr (Mode == segmented_scan_study::generation_mode::sampled)
  {
    generation_seed = segmented_scan_study::study_generation_seed(state);
  }
  const auto weights = segmented_scan_study::make_multimodal_weights(
    count, long_fraction, long_to_short_ratio, generation_seed, shuffle_seed, Mode);
  segmented_scan_study::run(state, weights);
}

void multimodal_sampled(nvbench::state& state)
{
  multimodal<segmented_scan_study::generation_mode::sampled>(state);
}

void multimodal_quantile(nvbench::state& state)
{
  multimodal<segmented_scan_study::generation_mode::quantile>(state);
}
} // namespace

NVBENCH_BENCH(multimodal_sampled)
  .set_name("multimodal_sampled")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.04})
  .add_float64_axis("LongToShortRatio{io}", {4.0, 32.0, 128.0})
  .add_int64_axis("GenerationSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

NVBENCH_BENCH(multimodal_quantile)
  .set_name("multimodal_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.04})
  .add_float64_axis("LongToShortRatio{io}", {4.0, 32.0, 128.0})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
