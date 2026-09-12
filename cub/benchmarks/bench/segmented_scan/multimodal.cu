// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "segments_base.cuh"

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

NVBENCH_BENCH_TYPES(multimodal_sampled, NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("sampled")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.05, 0.10, 0.20})
  .add_float64_axis("LongToShortRatio{io}", {2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0});

NVBENCH_BENCH_TYPES(multimodal_quantile, NVBENCH_TYPE_AXES(segmented_scan_study::value_types, segmented_scan_study::offset_types))
  .set_name("quantile")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {32, 64, 128, 256, 512, 1024, 2048})
  .add_float64_axis("LongSegmentFraction{io}", {0.02, 0.05, 0.10, 0.20})
  .add_float64_axis("LongToShortRatio{io}", {2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0});
