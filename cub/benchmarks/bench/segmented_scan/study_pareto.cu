// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "study_base.cuh"

namespace
{
template <segmented_scan_study::generation_mode Mode>
void pareto(nvbench::state& state)
{
  if (!segmented_scan_study::is_study_cell(state))
  {
    return;
  }
  const auto count   = segmented_scan_study::segment_count(state);
  const auto alpha   = state.get_float64("Alpha{io}");
  const auto weights = segmented_scan_study::make_pareto_weights(
    count, alpha, segmented_scan_study::study_seed(state), Mode);
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
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_float64_axis("Alpha{io}", {100.0, 2.1, 1.5, 1.25})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});

NVBENCH_BENCH(pareto_quantile)
  .set_name("pareto_quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {256, 512, 2048})
  .add_float64_axis("Alpha{io}", {100.0, 2.1, 1.5, 1.25})
  .add_int64_axis("Seed{io}", {1, 2, 3, 4, 5, 6, 7, 8, 9, 42});
