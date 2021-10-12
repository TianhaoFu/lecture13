/*!
 * \file needle/device_api.h
 * \brief A device api that abstracts away memory managements.
 */
#ifndef NEEDLE_DEVICE_API_H_
#define NEEDLE_DEVICE_API_H_

#include <needle/dlpack.h>

// whether build with cuda support
#ifndef NEEDLE_USE_CUDA
#define NEEDLE_USE_CUDA 1
#endif

namespace needle {

/*! \brief the device indicator. */
using Device = DLDevice;

/*! \brief device stream */
using DLStreamHandle = void *;

/*!
 *  \brief Runtime device API that abstracts away memory managements.
 */
class DeviceAPI {
public:
  /*! \brief virtual destructor */
  virtual ~DeviceAPI() {}
  /*!
   * \brief Set the environment device id to device
   * \param dev The device to be set.
   */
  virtual void SetDevice(Device dev) = 0;
  /*!
   * \brief Allocate a data space on device.
   * \param dev The device device to perform operation.
   * \param nbytes The number of bytes in memory.
   * \return The allocated device pointer.
   */
  virtual void *AllocDataSpace(Device dev, size_t nbytes) = 0;
  /*!
   * \brief Free a data space on device.
   * \param dev The device device to perform operation.
   * \param ptr The data space.
   */
  virtual void FreeDataSpace(Device dev, void *ptr) = 0;
  /*!
   * \brief Copy data from one place to another
   * \param from The source array.
   * \param to The target array.
   * \param num_bytes The size of the memory in bytes
   * \param dev_from The source device
   * \param dev_to The target device
   * \param stream Optional stream object.
   */
  virtual void CopyDataFromTo(const void *from, void *to, size_t num_bytes,
                              Device dev_from, Device dev_to,
                              DLStreamHandle stream = nullptr) = 0;
  /*!
   * \brief StreamSync the based on the device
   * \param dev The device to synchronize to
   * \param stream Additional stream
   */
  virtual void StreamSync(Device dev, DLStreamHandle stream = nullptr) = 0;

  /*!
   * \brief get the device API
   * \param dev The device.
   * \return the created device API.
   */
  static DeviceAPI *Get(Device dev);
};

} // namespace needle

#endif // NEEDLE_DEVICE_API_H_
