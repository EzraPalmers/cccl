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
  const auto count        = segmented_scan_study::segment_count(state);
  const auto exponent     = state.get_float64("Exponent{io}");
  const auto shuffle_seed = segmented_scan_study::study_shuffle_seed(state);
  segmented_scan_study::seed_type generation_seed{0};
  if constexpr (Mode == segmented_scan_study::generation_mode::sampled)
  {
    generation_seed = segmented_scan_study::study_generation_seed(state);
  }
  const auto weights =
    segmented_scan_study::make_zipf_weights(count, exponent, generation_seed, shuffle_seed, Mode);
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
  .set_name("sampled")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.25, 1.5, 1.75, 1.9})
  .add_int64_axis("GenerationSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

NVBENCH_BENCH(zipf_quantile)
  .set_name("quantile")
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", {128, 256, 512, 2048})
  .add_float64_axis("Exponent{io}", {0.0, 0.75, 1.25, 1.5, 1.75, 1.9})
  .add_int64_axis("ShuffleSeed{io}", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
