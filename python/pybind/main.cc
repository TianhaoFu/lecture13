/*!
 * Pybind module that exposes C/C++ object functions to the python side.
 *
 * Read more about pybind11 here https://pybind11.readthedocs.io/
 */

// This file exposes functions in include/needle to python interface
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <needle/cuda_ops.h>
#include <needle/logging.h>
#include <needle/ndarray.h>

#include <sstream>

namespace needle {

DLDevice CreateDLDevice(int device_type, int device_id) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  return dev;
}

} // namespace needle

PYBIND11_MODULE(main, m) {
  namespace py = pybind11;

  using namespace needle;
  m.doc() = "pybind11 bindings";

  py::class_<NDArray>(m, "NDArray")
      .def("shape",
           [](const NDArray &self) -> py::tuple {
             return py::cast(self.Shape());
           })
      .def("dtype",
           [](const NDArray &self) -> py::str {
             return DLDataType2String(self->dtype);
           })
      .def("dldevice",
           [](const NDArray &self) -> DLDevice { return self->device; })
      .def("copyfrombytes",
           [](NDArray self, py::array arr, size_t size) {
             py::buffer_info info = arr.request();
             self.CopyFromBytes(info.ptr, size);
           })
      .def("copytobytes",
           [](NDArray self, py::array arr, size_t size) {
             py::buffer_info info = arr.request();
             self.CopyToBytes(info.ptr, size);
           })
      .def("copyto", [](NDArray self, NDArray other) { self.CopyTo(other); });

  py::class_<DLDevice>(m, "DLDevice")
      .def(py::init(&CreateDLDevice))
      .def_readonly("device_type", &DLDevice::device_type)
      .def_readonly("device_id", &DLDevice::device_id);

  m.def("empty", [](ShapeTuple shape, std::string dtype, DLDevice device) {
    return NDArray::Empty(shape, String2DLDataType(dtype), device);
  });

#if NEEDLE_USE_CUDA
  m.def("CUDAFill", cuda::Fill);

  m.def("CUDAEWiseAdd", cuda::EWiseAdd);
  m.def("CUDAAddScalar", cuda::AddScalar);

  m.def("CUDAEWiseMul", cuda::EWiseMul);
  m.def("CUDAMulScalar", cuda::MulScalar);
#endif
}
