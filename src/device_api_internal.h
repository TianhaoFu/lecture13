/*!
 * \file device_api_internal.h
 * \brief Private internal header that delcares common parts among device.
 */
#ifndef NEEDLE_DEVICE_API_INTERNAL_H_
#define NEEDLE_DEVICE_API_INTERNAL_H_

#include <needle/device_api.h>
#include <needle/dlpack.h>

namespace needle {
/*! \return a global singleton of CPU device api */
DeviceAPI *GlobalCPUDeviceAPI();

/*! \return a global singleton of CUDA device api */
DeviceAPI *GlobalCUDADeviceAPI();
} // namespace needle
#endif // NEEDLE_DEVICE_API_INTERNAL_H_
