#include <needle/device_api.h>
#include <needle/logging.h>

#include "./device_api_internal.h"

namespace needle {

DeviceAPI *DeviceAPI::Get(Device device) {
  switch (device.device_type) {
  case kDLCPU:
    return GlobalCPUDeviceAPI();
  case kDLCUDA: {
#if NEEDLE_USE_CUDA
    return GlobalCUDADeviceAPI();
#else
    LOG(FATAL) << "CUDA is not enabled";
    return nullptr;
#endif
  }
  default:
    LOG(FATAL) << "Unsupported device_type"
               << static_cast<int>(device.device_type);
    return nullptr;
  };
}

} // namespace needle
