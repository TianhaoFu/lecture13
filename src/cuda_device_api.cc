/*!
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include "./device_api_internal.h"
#include <needle/device_api.h>
#include <needle/logging.h>

#if NEEDLE_USE_CUDA
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

namespace needle {

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

class CUDADeviceAPI final : public DeviceAPI {
public:
  void SetDevice(Device dev) final { CUDA_CALL(cudaSetDevice(dev.device_id)); }

  void *AllocDataSpace(Device dev, size_t nbytes) final {
    void *ret;
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaMalloc(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaFree(ptr));
  }

  void CopyDataFromTo(const void *from, void *to, size_t size, Device dev_from,
                      Device dev_to, DLStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id,
                            size, cu_stream);
      }
    } else if (dev_from.device_type == kDLCUDA &&
               dev_to.device_type == kDLCPU) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU &&
               dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void StreamSync(Device dev, DLStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

private:
  static void GPUCopy(const void *from, void *to, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream) {
    if (stream != nullptr) {
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }
};

DeviceAPI *GlobalCUDADeviceAPI() {
  // singleton pattern
  static auto *inst = new CUDADeviceAPI();
  return inst;
}
} // namespace needle
#endif // NEEDLE_USE_CUDA
