// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "study_base.cuh"

namespace
{
template <segmented_scan_study::generation_mode Mode>
void zipf(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count    = segmented_scan_study::segment_count(state);
  const auto exponent = state.get_float64("Exponent{io}");
  const auto weights  = segmented_scan_study::make_zipf_weights(
    count, exponent, segmented_scan_study::study_seed(state), Mode);
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
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.5, 2.2, 3.0})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});

NVBENCH_BENCH(zipf_quantile)
  .set_name("zipf_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.5, 2.2, 3.0})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});
