#include <needle/device_api.h>
#include <needle/logging.h>

#include <cstring>

#include "./device_api_internal.h"

namespace needle {

class CPUDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(Device dev) final {}

  void *AllocDataSpace(Device dev, size_t nbytes) final {
    // align to 256 bytes
    size_t alignment = 256;

    void *ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr)
      throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0)
      throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(Device dev, void *ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void StreamSync(Device dev, DLStreamHandle stream) final {}

  void CopyDataFromTo(const void *from, void *to, size_t size, Device dev_from,
                      Device dev_to, DLStreamHandle stream) final {
    CHECK_EQ(dev_from.device_type, kDLCPU);
    CHECK_EQ(dev_to.device_type, kDLCPU);
    memcpy(to, from, size);
  }
};

DeviceAPI *GlobalCPUDeviceAPI() {
  // singleton pattern
  static auto *inst = new CPUDeviceAPI();
  return inst;
}

} // namespace needle
