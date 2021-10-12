"""A implementation of NDArray"""
import numpy as np
import needle._ffi
from needle.device import BackendNDArrayBase, DLDeviceType


class NDArray(BackendNDArrayBase):
    """A NDArray backed by the C++ array """

    handle: needle._ffi.NDArray

    def __init__(self, handle):
        self.handle = handle

    @property
    def shape(self):
        return self.handle.shape()

    @property
    def dtype(self):
        return self.handle.dtype()

    @property
    def dldevice(self):
        return self.handle.dldevice()

    def __repr__(self):
        return self.numpy().__repr__()

    def __str__(self):
        return self.numpy().__str__()

    def copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArray):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError(
                    "array must be an array_like data,"
                    + "type %s is not supported" % str(type(source_array))
                )

        shape, dtype = self.shape, self.dtype
        if source_array.shape != shape:
            raise ValueError(
                "array shape do not match the shape of NDArray {0} vs {1}".format(
                    source_array.shape, shape
                )
            )
        source_array = np.ascontiguousarray(source_array, dtype)
        assert source_array.flags["C_CONTIGUOUS"]
        nbytes = source_array.size * source_array.dtype.itemsize
        self.handle.copyfrombytes(source_array, nbytes)
        return self

    def numpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        shape, dtype = self.shape, self.dtype
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags["C_CONTIGUOUS"]
        nbytes = np_arr.size * np_arr.dtype.itemsize
        self.handle.copytobytes(np_arr, nbytes)
        return np_arr

    def copyto(self, target):
        """Copy array to target

        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, NDArray):
            self.handle.copyto(target.handle)
            return target

        if isinstance(target, needle._ffi.DLDevice):
            res = empty(self.shape, self.dtype, target)
            self.handle.copyto(res.handle)
            return res

        raise ValueError("Unsupported target type %s" % str(type(target)))


def dldevice(device_type, device_id):
    return needle._ffi.DLDevice(device_type, device_id)


def cpu(device_id=0):
    return dldevice(DLDeviceType.CPU, 0)


def cuda(device_id=0):
    return dldevice(DLDeviceType.CUDA, 0)


def empty(shape, dtype=None, dldevice=None):
    dtype = "float32" if dtype is None else str(dtype)
    dldevice = cpu() if dldevice is None else dldevice
    return NDArray(needle._ffi.empty(shape, dtype, dldevice))


def array(arr, dtype=None, dldevice=None):
    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr)
    dtype = dtype if dtype else arr.dtype
    return empty(arr.shape, dtype, dldevice).copyfrom(arr)


def empty_like(arr):
    return empty(arr.shape, dtype=arr.dtype, dldevice=arr.dldevice)
