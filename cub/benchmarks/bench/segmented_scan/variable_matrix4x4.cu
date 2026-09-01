// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

struct matrix4x4
{
  cuda::std::uint32_t values[16];

  _CCCL_HOST_DEVICE_API constexpr matrix4x4() noexcept
      : values{}
  {
    values[0]  = 1;
    values[5]  = 1;
    values[10] = 1;
    values[15] = 1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr matrix4x4 zero() noexcept
  {
    matrix4x4 result{};
    for (auto& value : result.values)
    {
      value = 0;
    }
    return result;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool operator==(const matrix4x4& other) const noexcept
  {
    for (std::size_t index = 0; index < 16; ++index)
    {
      if (values[index] != other.values[index])
      {
        return false;
      }
    }
    return true;
  }
};

static_assert(sizeof(matrix4x4) == 64);
static_assert(cuda::std::is_default_constructible_v<matrix4x4>);
static_assert(cuda::std::is_trivially_copyable_v<matrix4x4>);

NVBENCH_DECLARE_TYPE_STRINGS(matrix4x4, "M4x4U32", "matrix4x4<uint32_t>");

struct matrix_multiply
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr matrix4x4
  operator()(const matrix4x4& left, const matrix4x4& right) const noexcept
  {
    auto result = matrix4x4::zero();
    for (std::size_t row = 0; row < 4; ++row)
    {
      for (std::size_t column = 0; column < 4; ++column)
      {
        for (std::size_t inner = 0; inner < 4; ++inner)
        {
          result.values[row * 4 + column] += left.values[row * 4 + inner] * right.values[inner * 4 + column];
        }
      }
    }
    return result;
  }
};

namespace
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr cuda::std::uint64_t matrix_counter_u64(
  cuda::std::uint64_t index) noexcept
{
  auto value = 0x4D41545249583434ULL ^ (index * 0x9E3779B97F4A7C15ULL);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

struct make_matrix
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr matrix4x4 operator()(cuda::std::uint64_t index) const noexcept
  {
    auto result = matrix4x4::zero();
    for (cuda::std::uint64_t cell = 0; cell < 16; ++cell)
    {
      result.values[cell] = static_cast<cuda::std::uint32_t>((matrix_counter_u64(index * 16 + cell) & 3) + 1);
    }
    return result;
  }
};

struct matrix_benchmark_traits
{
  using value_types = nvbench::type_list<matrix4x4>;

  template <typename T, typename OffsetT>
  [[nodiscard]] static thrust::device_vector<T> make_input(OffsetT elements)
  {
    static_assert(cuda::std::is_same_v<T, matrix4x4>);
    auto input = thrust::device_vector<T>(elements, thrust::default_init);
    thrust::tabulate(input.begin(), input.end(), make_matrix{});
    return input;
  }

  template <typename OffsetT>
  static void validate(const thrust::device_vector<matrix4x4>& input,
                       const thrust::device_vector<matrix4x4>& output,
                       const thrust::device_vector<OffsetT>& offsets)
  {
    const thrust::host_vector<OffsetT> host_offsets = offsets;
    const auto num_segments = static_cast<OffsetT>(host_offsets.size() - 1);

    OffsetT longest_segment        = 0;
    OffsetT longest_segment_length = 0;
    for (OffsetT segment = 0; segment < num_segments; ++segment)
    {
      const auto segment_length = host_offsets[segment + 1] - host_offsets[segment];
      if (segment_length > longest_segment_length)
      {
        longest_segment        = segment;
        longest_segment_length = segment_length;
      }
    }

    std::vector<OffsetT> sampled_segments;
    const auto add_segment = [&](OffsetT segment) {
      if (std::find(sampled_segments.begin(), sampled_segments.end(), segment) == sampled_segments.end())
      {
        sampled_segments.push_back(segment);
      }
    };
    add_segment(OffsetT{0});
    add_segment(num_segments / 2);
    add_segment(num_segments - 1);
    add_segment(longest_segment);

    for (const auto segment : sampled_segments)
    {
      const auto begin       = host_offsets[segment];
      const auto segment_end = host_offsets[segment + 1];
      const auto sample_end  = begin + cuda::std::min(OffsetT{257}, segment_end - begin);
      const thrust::host_vector<matrix4x4> host_input(input.begin() + begin, input.begin() + sample_end);
      const thrust::host_vector<matrix4x4> host_output(output.begin() + begin, output.begin() + sample_end);

      auto reference = matrix4x4{};
      for (std::size_t prefix = 0; prefix < host_input.size(); ++prefix)
      {
        if (!(host_output[prefix] == reference))
        {
          throw std::runtime_error(
            "matrix validation failed at segment " + std::to_string(segment) + ", prefix " + std::to_string(prefix));
        }
        reference = matrix_multiply{}(reference, host_input[prefix]);
      }
    }

    const auto longest_begin = host_offsets[longest_segment];
    const auto tail_end      = host_offsets[longest_segment + 1];
    const auto tail_begin    = tail_end - cuda::std::min(OffsetT{257}, longest_segment_length);
    const thrust::host_vector<matrix4x4> host_input(input.begin() + tail_begin, input.begin() + tail_end);
    const thrust::host_vector<matrix4x4> host_output(output.begin() + tail_begin, output.begin() + tail_end);

    auto reference = matrix4x4{};
    if (tail_begin != longest_begin)
    {
      const matrix4x4 preceding_output = output[tail_begin - 1];
      const matrix4x4 preceding_input  = input[tail_begin - 1];
      reference                        = matrix_multiply{}(preceding_output, preceding_input);
    }
    for (std::size_t prefix = 0; prefix < host_input.size(); ++prefix)
    {
      if (!(host_output[prefix] == reference))
      {
        throw std::runtime_error(
          "matrix validation failed at segment " + std::to_string(longest_segment) + ", prefix "
          + std::to_string(static_cast<std::size_t>(tail_begin - longest_begin) + prefix));
      }
      reference = matrix_multiply{}(reference, host_input[prefix]);
    }
  }
};
} // namespace

using op_t = matrix_multiply;

#define VARIABLE_BENCHMARK_TRAITS matrix_benchmark_traits
#include "variable_base.cuh"
#undef VARIABLE_BENCHMARK_TRAITS
