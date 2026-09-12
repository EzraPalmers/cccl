// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "variable_base.cuh"

namespace
{
template <segmented_scan_study::generation_mode Mode, typename T>
void lognormal(nvbench::state& state)
{
  const auto count   = segmented_scan_study::segment_count(state);
  const auto sigma   = state.get_float64("Sigma{io}");
  const auto weights = segmented_scan_study::make_lognormal_weights(
    count, sigma, segmented_scan_study::fixed_generation_seed, segmented_scan_study::fixed_shuffle_seed, Mode);
  segmented_scan_study::run<T>(state, weights);
}

template <typename T, typename OffsetT>
void lognormal_sampled(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  lognormal<segmented_scan_study::generation_mode::sampled, T>(state);
}

template <typename T, typename OffsetT>
void lognormal_quantile(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  lognormal<segmented_scan_study::generation_mode::quantile, T>(state);
}
} // namespace

NVBENCH_BENCH_TYPES(lognormal_sampled,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("lognormal_sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Sigma{io}", {0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8,  0.85, 0.9,  0.95, 1.0,
                                  1.05, 1.1,  1.15, 1.2,  1.25, 1.3,  1.35, 1.4, 1.45, 1.5,  1.55, 1.6,  1.65,
                                  1.7,  1.75, 1.8,  1.85, 1.9,  1.95, 2.0,  2.1, 2.2,  2.3,  2.4,  2.5});

NVBENCH_BENCH_TYPES(lognormal_quantile,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("lognormal_quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Sigma{io}", {0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8,  0.85, 0.9,  0.95, 1.0,
                                  1.05, 1.1,  1.15, 1.2,  1.25, 1.3,  1.35, 1.4, 1.45, 1.5,  1.55, 1.6,  1.65,
                                  1.7,  1.75, 1.8,  1.85, 1.9,  1.95, 2.0,  2.1, 2.2,  2.3,  2.4,  2.5});

namespace
{
template <segmented_scan_study::generation_mode Mode, typename T>
void pareto(nvbench::state& state)
{
  const auto count   = segmented_scan_study::segment_count(state);
  const auto alpha   = state.get_float64("Alpha{io}");
  const auto weights = segmented_scan_study::make_pareto_weights(
    count, alpha, segmented_scan_study::fixed_generation_seed, segmented_scan_study::fixed_shuffle_seed, Mode);
  segmented_scan_study::run<T>(state, weights);
}

template <typename T, typename OffsetT>
void pareto_sampled(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  pareto<segmented_scan_study::generation_mode::sampled, T>(state);
}

template <typename T, typename OffsetT>
void pareto_quantile(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  pareto<segmented_scan_study::generation_mode::quantile, T>(state);
}
} // namespace

NVBENCH_BENCH_TYPES(pareto_sampled,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("pareto_sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Alpha{io}", {1.3,  1.4, 1.5,  1.55, 1.6,  1.65, 1.7,  1.75, 1.8,  1.85, 1.9, 1.95, 2.0,
                                  2.05, 2.1, 2.15, 2.2,  2.25, 2.3,  2.35, 2.4,  2.45, 2.5,  2.6, 2.7,  2.8,
                                  2.9,  3.0, 3.2,  3.4,  3.6,  3.8,  4.0,  4.2,  4.4,  4.6,  4.8, 5.0});

NVBENCH_BENCH_TYPES(pareto_quantile,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("pareto_quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Alpha{io}", {1.3,  1.4, 1.5,  1.55, 1.6,  1.65, 1.7,  1.75, 1.8,  1.85, 1.9, 1.95, 2.0,
                                  2.05, 2.1, 2.15, 2.2,  2.25, 2.3,  2.35, 2.4,  2.45, 2.5,  2.6, 2.7,  2.8,
                                  2.9,  3.0, 3.2,  3.4,  3.6,  3.8,  4.0,  4.2,  4.4,  4.6,  4.8, 5.0});

namespace
{
template <segmented_scan_study::generation_mode Mode, typename T>
void zipf(nvbench::state& state)
{
  const auto count    = segmented_scan_study::segment_count(state);
  const auto exponent = state.get_float64("Exponent{io}");
  const auto weights  = segmented_scan_study::make_zipf_weights(
    count, exponent, segmented_scan_study::fixed_generation_seed, segmented_scan_study::fixed_shuffle_seed, Mode);
  segmented_scan_study::run<T>(state, weights);
}

template <typename T, typename OffsetT>
void zipf_sampled(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  zipf<segmented_scan_study::generation_mode::sampled, T>(state);
}

template <typename T, typename OffsetT>
void zipf_quantile(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  zipf<segmented_scan_study::generation_mode::quantile, T>(state);
}
} // namespace

NVBENCH_BENCH_TYPES(zipf_sampled,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("zipf_sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Exponent{io}", {0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.75, 0.8,  0.85, 0.9,  0.95,
                                     1.0,  1.05, 1.1,  1.15, 1.2,  1.25, 1.3,  1.35, 1.4,  1.45, 1.5,  1.55, 1.6,
                                     1.65, 1.7,  1.75, 1.8,  1.85, 1.9,  1.95, 2.0,  2.2,  2.5,  3.0});

NVBENCH_BENCH_TYPES(zipf_quantile,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("zipf_quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("Exponent{io}", {0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.75, 0.8,  0.85, 0.9,  0.95,
                                     1.0,  1.05, 1.1,  1.15, 1.2,  1.25, 1.3,  1.35, 1.4,  1.45, 1.5,  1.55, 1.6,
                                     1.65, 1.7,  1.75, 1.8,  1.85, 1.9,  1.95, 2.0,  2.2,  2.5,  3.0});

namespace
{
template <segmented_scan_study::generation_mode Mode, typename T>
void multimodal(nvbench::state& state)
{
  const auto count               = segmented_scan_study::segment_count(state);
  const auto long_fraction       = state.get_float64("LongSegmentFraction{io}");
  const auto long_to_short_ratio = state.get_float64("LongToShortRatio{io}");
  const auto weights             = segmented_scan_study::make_multimodal_weights(
    count,
    long_fraction,
    long_to_short_ratio,
    segmented_scan_study::fixed_generation_seed,
    segmented_scan_study::fixed_shuffle_seed,
    Mode);
  segmented_scan_study::run<T>(state, weights);
}

template <typename T, typename OffsetT>
void multimodal_sampled(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  multimodal<segmented_scan_study::generation_mode::sampled, T>(state);
}

template <typename T, typename OffsetT>
void multimodal_quantile(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  multimodal<segmented_scan_study::generation_mode::quantile, T>(state);
}
} // namespace

NVBENCH_BENCH_TYPES(multimodal_sampled,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("multimodal_sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.05, 0.10, 0.20})
  .add_float64_axis("LongToShortRatio{io}", {2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0});

NVBENCH_BENCH_TYPES(multimodal_quantile,
                    NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("multimodal_quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.05, 0.10, 0.20})
  .add_float64_axis("LongToShortRatio{io}", {2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0});
