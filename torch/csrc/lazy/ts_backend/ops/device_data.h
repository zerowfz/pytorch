#pragma once

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  static std::vector<BackendDataPtr> backend_data_storage;
  static std::unordered_map<BackendData::Handle, size_t> backend_data_handle_map;

  explicit DeviceData(BackendDataPtr data);

  std::string ToString() const override;

  const BackendDataPtr& data() const {
    std::cout << "Read from DeviceData::backend_data_storage[" << index_ << "]" << std::endl;
    // This is only valid during tracing time. After each tracing,
    // we move backend_data_storage to the exection thread.
    TORCH_CHECK(index_ < backend_data_storage.size(), "index_: ", index_, ", backend_data_storage.size(): ", backend_data_storage.size());
    return backend_data_storage[index_];
  }

  size_t Index() const {
    return index_;
  }

  static const DeviceData* Cast(const Node* node);

  static void ResetBackendDataStorage();

 private:
  size_t ComputeOrGetIndex(BackendDataPtr data);

  size_t index_;
};

} // namespace lazy
} // namespace torch
