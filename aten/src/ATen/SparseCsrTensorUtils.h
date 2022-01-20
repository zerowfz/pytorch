#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace sparse_csr {

using SparseCsrTensor = Tensor;

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  AT_ASSERTM(
      self.is_sparse_csr(),
      "_internal_get_SparseCsrTensorImpl: not a sparse CSR tensor");
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}

inline void _validate_sparse_csr_tensor_args(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size,
    bool is_block_sparse) {
  // Layout Invariants
  TORCH_CHECK(
      col_indices.layout() == kStrided && col_indices.is_contiguous(),
      "expected col_indices to be a strided and contiguous tensor");

  TORCH_CHECK(
      crow_indices.layout() == kStrided && crow_indices.is_contiguous(),
      "expected crow_indices to be a strided and contiguous tensor");

  TORCH_CHECK(
      values.layout() == kStrided && values.is_contiguous(),
      "expected values to be a strided and contiguous tensor");

  // Shape and Strides invariants
  TORCH_CHECK(
      size.size() == 2,
      "dimension of a CSR tensor must be 2, but got: ",
      size.size());
  TORCH_CHECK(
      crow_indices.dim() == 1,
      "crow_indices must have dim=1 but got crow_indices.dim()=",
      crow_indices.dim());
  TORCH_CHECK(
      col_indices.dim() == 1,
      "col_indices must have dim=1 but got col_indices.dim()=",
      col_indices.dim());
  TORCH_CHECK(
      values.dim() == 1 || is_block_sparse,
      "values must have dim=1 or dim=3 but got values.dim()=",
      values.dim());
  int64_t blocksize[2];
  if (is_block_sparse) {
    TORCH_CHECK(
        values.device().type() == kCPU,
        "device type of blocksparse values (",
        values.device().type(),
        ") must be CPU ",
        "but got ",
        values.device(),
        "instead.");
    blocksize[0] = values.size(1);
    blocksize[1] = values.size(2);
    int64_t block_numel = blocksize[0] * blocksize[1];
    TORCH_CHECK(
        blocksize[0] == blocksize[1] && blocksize[0] > 1,
        "For block sparse CSR Tensors (3-dim values) the ",
        "blocks must be square and greater than 1. ",
        "Got (",
        blocksize[0],
        ", ",
        blocksize[1],
        ") instead.");
    TORCH_CHECK(
        blocksize[0] == blocksize[1] && blocksize[0] > 1,
        "Block sparse CSR Tensors must have a size that is an integral multiple of their block size.",
        "Got (",
        size[0],
        ", ",
        size[1],
        ") with block size (",
        blocksize[0],
        ", ",
        blocksize[1],
        ") instead.");
    TORCH_CHECK(
        crow_indices.numel() == (size[0] / blocksize[0] + 1),
        "crow_indices.numel() must be size(0) / size(1) + 1, but got: ",
        crow_indices.numel());
    TORCH_CHECK(
        col_indices.numel() == values.size(0),
        "col_indices and values leading size must have be equal, but got col_indices.numel(): ",
        col_indices.numel(),
        ", values.size(0): ",
        values.size(0));
  } else {
    // Note, this check also enforces `crow_indices.numel() >= 1`
    TORCH_CHECK(
        crow_indices.numel() == (size[0] + 1),
        "crow_indices.numel() must be size(0) + 1, but got: ",
        crow_indices.numel());
    TORCH_CHECK(
        col_indices.numel() == values.numel(),
        "col_indices and values must have equal sizes, but got col_indices.numel(): ",
        col_indices.numel(),
        ", values.numel(): ",
        values.numel());
  }

  // Indices invariants
  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
    Tensor crow_indices_cpu = crow_indices.to(kCPU);
    auto crow_indices_accessor = crow_indices_cpu.accessor<index_t, 1>();
    TORCH_CHECK(
        crow_indices_accessor[0] == 0, "0th value of crow_indices must be 0.");

    TORCH_CHECK(
        crow_indices_accessor[crow_indices.numel() - 1] == col_indices.numel(),
        "last value of crow_indices should be equal to the length of col_indices.");

    for (int i = 1; i <= size[0]; i++) {
      TORCH_CHECK(
          crow_indices_accessor[i - 1] <= crow_indices_accessor[i],
          "at position i = ",
          i,
          ", this condition crow_indices[i - 1] <= crow_indices[i] fails");
    }
    if (col_indices.numel() > 0) {
      TORCH_CHECK(
          0 <= col_indices.min().item<index_t>(),
          "col_indices.min() should be greater or equal to zero");
      if (is_block_sparse) {
        TORCH_CHECK(
            (size[1] / blocksize[0]) > col_indices.max().item<index_t>(),
            "size(1) / values.size(0) should be greater than col_indices.max()");
      } else {
        TORCH_CHECK(
            size[1] > col_indices.max().item<index_t>(),
            "size(1) should be greater than col_indices.max()");
      }
    }
  });

  // CSR Type Invariants
  auto crow_indices_type = crow_indices.scalar_type();
  auto col_indices_type = col_indices.scalar_type();
  TORCH_CHECK(
      crow_indices_type == col_indices_type,
      "both crow_indices and col_indices should have the same type.");
  TORCH_CHECK(
      crow_indices_type == kInt || crow_indices_type == kLong,
      "crow_indices and col_indices must be an int32 or int64 type, but got: ",
      crow_indices_type);

  // CSR Device Invariants
  TORCH_CHECK(
      col_indices.get_device() == crow_indices.get_device(),
      "crow_indices and col_indices devices (",
      crow_indices.get_device(),
      ", ",
      col_indices.get_device(),
      ") must match");
  TORCH_CHECK(
      crow_indices.get_device() == values.get_device(),
      "device of crow_indices (",
      crow_indices.get_device(),
      ") must match device of values (",
      values.get_device(),
      ")");
  TORCH_CHECK(
      values.device().type() == kCPU || values.device().type() == kCUDA,
      "device type of values (",
      values.device().type(),
      ") must be CPU or CUDA ",
      "but got ",
      values.device(),
      "instead.");
}

inline void _validate_sparse_csr_tensor_args(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size) {
  bool is_block_sparse = (values.dim() == 3);
  _validate_sparse_csr_tensor_args(
      crow_indices, col_indices, values, size, is_block_sparse);
}

inline void _validate_sparse_csr_tensor_args(
    const Tensor& input, // Must be a CSR Tensor
    bool is_block_sparse) {
  _validate_sparse_csr_tensor_args(
      input.crow_indices(),
      input.col_indices(),
      input.values(),
      input.sizes(),
      is_block_sparse);
}

inline void _validate_sparse_csr_tensor_args(
    const Tensor& input // Must be a CSR Tensor
) {
  bool is_block_sparse = (input.values().dim() == 3);
  _validate_sparse_csr_tensor_args(input, is_block_sparse);
}

} // namespace sparse_csr
} // namespace at
