#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  explicit DeviceData(std::shared_ptr<BackendData> data);
  explicit DeviceData(std::shared_ptr<BackendData> data, hash_t hash_seed);

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  static const DeviceData* Cast(const Node* node);

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch
