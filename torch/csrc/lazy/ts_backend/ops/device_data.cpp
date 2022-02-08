#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

#include <sstream>

namespace torch {
namespace lazy {

std::vector<BackendDataPtr> DeviceData::backend_data_storage{};

std::unordered_map<BackendData::Handle, size_t> DeviceData::backend_data_handle_map;

DeviceData::DeviceData(BackendDataPtr data)
    : TsNode(
          ltc_device_data,
          data->shape(),
          /*num_outputs=*/1,
          /*Use index_ as hash_seed*/
          ComputeOrGetIndex(data)),
      index_(ComputeOrGetIndex(data)) {
    std::cout << "Write into DeviceData::backend_data_storage[" << index_ << "]" << std::endl ;
}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data()->device();
  return ss.str();
}

const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, ltc_device_data);
}

size_t DeviceData::ComputeOrGetIndex(BackendDataPtr data) {
  auto it = backend_data_handle_map.find(data->GetHandle());
  if (it != backend_data_handle_map.end()) {
    return it->second;
  } else {
    size_t index = backend_data_storage.size();
    backend_data_handle_map[data->GetHandle()] = index;
    backend_data_storage.push_back(data);
    return index;
  }
}

void DeviceData::ResetBackendDataStorage() {
  backend_data_storage.clear();
  backend_data_handle_map.clear();

  static size_t iteration = 0;
  std::cout << "Done with tracing iteration " << iteration++ << std::endl << std::endl;
}

} // namespace lazy
} // namespace torch
