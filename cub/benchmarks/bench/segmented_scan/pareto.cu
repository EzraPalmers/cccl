// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "segments_base.cuh"

namespace
{
template <segmented_scan_study::generation_mode Mode, typename T>
void pareto(nvbench::state& state)
{
  const auto count = segmented_scan_study::segment_count(state);
  const auto alpha  = state.get_float64("Alpha{io}");
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

NVBENCH_BENCH_TYPES(pareto_sampled, NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis(
    "Alpha{io}",
    {1.3, 1.4,
     1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4,
     2.45, 2.5,
     2.6, 2.7, 2.8, 2.9, 3.0,
     3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0});

NVBENCH_BENCH_TYPES(pareto_quantile, NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis(
    "Alpha{io}",
    {1.3, 1.4,
     1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4,
     2.45, 2.5,
     2.6, 2.7, 2.8, 2.9, 3.0,
     3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0});
