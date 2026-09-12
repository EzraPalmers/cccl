// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "variable_base.cuh"

namespace
{
// Even segment lengths, no distribution: measures the under-filling curve alone, against which
// the ragged families' additional imbalance loss is read.
template <typename T, typename OffsetT>
void uniform(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto count   = segmented_scan_study::segment_count(state);
  const auto weights = thrust::device_vector<double>(count, 1.0);
  segmented_scan_study::run<T>(state, weights);
}
} // namespace

using uniform_value_types  = nvbench::type_list<::cuda::std::int32_t>;
using uniform_offset_types = nvbench::type_list<::cuda::std::int32_t>;

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(uniform_value_types, uniform_offset_types))
  .set_name("uniform")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {22, 26})
  .add_int64_axis("MeanSegmentSize{io}", nvbench::range(4, 2048, 4));
