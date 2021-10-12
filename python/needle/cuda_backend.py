"""CUDA computation backend.

This backend uses cuda backend_ndarray for cached data and redirects
calls to cuda kernel invocations.
"""
import needle._ffi
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr


class CUDADevice(Device):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def dldevice(self):
        return nd.dldevice(DLDeviceType.CUDA, self.device_id)

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __str__(self):
        return self.__repr__()

    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, dldevice=self.dldevice())

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, dldevice=self.dldevice())

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        needle._ffi.CUDAFill(fill_value, array.handle)
        return array

    def enabled(self):
        return hasattr(needle._ffi, "CUDAFill")

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.cuda_compute(inputs, attrs)


def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def register_cuda_compute(name, value=None):
    """Register the numpy compute property"""
    return register_op_attr(name, "cuda_compute", value)


# device specific computations
@register_cuda_compute("EWiseAdd")
def add(inputs, attrs):
    out = nd.empty_like(inputs[0])
    needle._ffi.CUDAEWiseAdd(inputs[0].handle, inputs[1].handle, out.handle)
    return out


@register_cuda_compute("AddScalar")
def add_scalar(inputs, attrs):
    out = nd.empty_like(inputs[0])
    needle._ffi.CUDAAddScalar(inputs[0].handle, attrs["scalar"], out.handle)
    return out


@register_cuda_compute("EWiseMul")
def mul(inputs, attrs):
    assert len(inputs) == 2
    out = nd.empty_like(inputs[0])
    needle._ffi.CUDAEWiseMul(inputs[0].handle, inputs[1].handle, out.handle)
    return out


@register_cuda_compute("MulScalar")
def mul(inputs, attrs):
    assert len(inputs) == 1
    out = nd.empty_like(inputs[0])
    needle._ffi.CUDAMulScalar(inputs[0].handle, attrs["scalar"], out.handle)
    return out
