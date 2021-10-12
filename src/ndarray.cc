/*!
 * \file ndarray.cc
 * \brief Implementation of NDArray
 */
#include <needle/device_api.h>
#include <needle/logging.h>
#include <needle/ndarray.h>

#include <memory>

namespace needle {

size_t NDArray::GetNumElements(const DLTensor *arr) {
  size_t size = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    size *= static_cast<size_t>(arr->shape[i]);
  }
  return size;
}

size_t NDArray::GetDataSize(const DLTensor *arr) {
  size_t size = GetNumElements(arr);
  int type_nbytes = (arr->dtype.bits * arr->dtype.lanes + 7) / 8;
  return size * type_nbytes;
}

bool NDArray::IsCompact(const DLTensor *arr) {
  if (arr->strides != nullptr) {
    int64_t expected_stride = 1;
    for (int32_t i = arr->ndim; i != 0; --i) {
      int32_t k = i - 1;
      if (arr->strides[k] != expected_stride) {
        return false;
      }
      expected_stride *= arr->shape[k];
    }
  }
  return arr->byte_offset == 0;
}

NDArray::Container::~Container() {
  // free the data associated with the container if any.
  if (dl_tensor.data != nullptr) {
    DeviceAPI::Get(dl_tensor.device)
        ->FreeDataSpace(dl_tensor.device, dl_tensor.data);
  }
}

NDArray NDArray::Empty(ShapeTuple shape, DLDataType dtype, Device device) {
  std::shared_ptr<Container> data = std::make_shared<Container>();
  data->shape_ = std::move(shape);
  data->dl_tensor.shape = data->shape_.data();
  data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
  data->dl_tensor.dtype = dtype;
  data->dl_tensor.device = device;

  // allocate data from the corresponding device api
  data->dl_tensor.data = DeviceAPI::Get(device)->AllocDataSpace(
      device, GetDataSize(&(data->dl_tensor)));
  return NDArray(data);
}

void NDArray::CopyFromTo(const DLTensor *from, DLTensor *to,
                         DLStreamHandle stream) {
  size_t from_size = GetDataSize(from);
  size_t to_size = GetDataSize(to);
  CHECK(IsCompact(from));
  CHECK(IsCompact(to));
  CHECK_EQ(from_size, to_size) << "The size must exactly match";

  CHECK(from->device.device_type == to->device.device_type ||
        from->device.device_type == kDLCPU || to->device.device_type == kDLCPU)
      << "Can not copy across different device types directly";

  // Use the device that is *not* a cpu device to get the correct device
  // api manager.
  Device dev = from->device.device_type != kDLCPU ? from->device : to->device;
  DeviceAPI::Get(dev)->CopyDataFromTo(from->data, to->data, from_size,
                                      from->device, to->device, stream);
}

void NDArray::CopyFromBytes(const void *data, size_t nbytes) {
  DLTensor *handle = &(data_->dl_tensor);

  DLTensor from;
  from.data = const_cast<void *>(data);
  from.device = Device{kDLCPU, 0};
  from.ndim = handle->ndim;
  from.dtype = handle->dtype;
  from.shape = handle->shape;
  from.strides = nullptr;
  from.byte_offset = 0;
  this->CopyFrom(&from);
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}

void NDArray::CopyToBytes(void *data, size_t nbytes) const {
  DLTensor *handle = &(data_->dl_tensor);
  DLTensor to;
  to.data = data;
  to.device = Device{kDLCPU, 0};
  to.ndim = handle->ndim;
  to.dtype = handle->dtype;
  to.shape = handle->shape;
  to.strides = nullptr;
  to.byte_offset = 0;
  this->CopyTo(&to);
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}

// dtype conversion.
inline const char *DLDataTypeCode2Str(DLDataTypeCode type_code) {
  switch (static_cast<int>(type_code)) {
  case kDLInt:
    return "int";
  case kDLUInt:
    return "uint";
  case kDLFloat:
    return "float";
  case kDLBfloat:
    return "bfloat";
  default:
    LOG(FATAL) << "unknown type_code=" << static_cast<int>(type_code);
    return "";
  }
}

inline std::ostream &operator<<(std::ostream &os, DLDataType t) { // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool";
    return os;
  }
  os << DLDataTypeCode2Str(static_cast<DLDataTypeCode>(t.code));
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}

std::string DLDataType2String(DLDataType t) {
  if (t.bits == 0)
    return "";
  std::ostringstream os;
  os << t;
  return os.str();
}

DLDataType String2DLDataType(std::string s) {
  DLDataType t;
  t.bits = 32;
  t.lanes = 1;
  const char *scan = "";

  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;
    scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt;
    scan = s.c_str() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat;
    scan = s.c_str() + 5;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  char *xdelim;
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0)
    t.bits = bits;
  char *endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, &endpt, 10));
  }
  CHECK(endpt == s.c_str() + s.length()) << "unknown type " << s;
  return t;
}

} // namespace needle
