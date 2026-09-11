// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "study_base.cuh"

#include <string>

namespace
{
template <segmented_scan_study::generation_mode Mode>
void multimodal(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto shape    = state.get_string("MultimodalShape{io}");
  const auto colon    = shape.find(':');
  if (colon == std::string::npos)
  {
    state.skip("unknown multimodal shape");
    return;
  }
  const auto long_fraction       = std::stod(shape.substr(0, colon));
  const auto long_to_short_ratio = std::stod(shape.substr(colon + 1));

  const auto count = segmented_scan_study::segment_count(state);
  const auto weights = segmented_scan_study::make_multimodal_weights(
    count,
    long_fraction,
    long_to_short_ratio,
    segmented_scan_study::study_seed(state),
    Mode);
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
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_string_axis("MultimodalShape{io}", {"0.05:2", "0.05:8", "0.05:32", "0.05:100", "0.05:200"})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});

NVBENCH_BENCH(multimodal_quantile)
  .set_name("multimodal_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_string_axis("MultimodalShape{io}", {"0.05:2", "0.05:8", "0.05:32", "0.05:100", "0.05:200"})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});
