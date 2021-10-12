/*!
 * \file needle/ndarray.h
 * \brief A device-independent managed NDArray abstraction.
 */
#ifndef NEEDLE_NDARRAY_H_
#define NEEDLE_NDARRAY_H_

#include <memory>
#include <needle/device_api.h>
#include <needle/dlpack.h>
#include <vector>

namespace needle {

/*! \brief Container to store the shape of the array. */
using ShapeTuple = std::vector<int64_t>;

/*!
 * \brief A C++ NDArray implementation.
 *
 * This class holds a reference counted pointer to the container object.
 * This allows us to hold multiple shallow copies of NDArray through a common
 * shared pointer. The NDArray won't be destructed until all the
 * references are de-allocated.
 *
 * \sa NDArray::Container
 */
class NDArray {
public:
  /*!
   * \brief Container class that defines a managed NDArray data structure.
   */
  class Container {
  public:
    /*! \brief DLTensor representation of the array */
    DLTensor dl_tensor;

    /*! \brief default constructor */
    Container() {
      dl_tensor.data = nullptr;
      dl_tensor.ndim = 0;
      dl_tensor.shape = nullptr;
      dl_tensor.strides = nullptr;
      dl_tensor.byte_offset = 0;
    }
    ~Container();

  private:
    /*! \brief Additional vector to store the shape */
    ShapeTuple shape_;
    /*! \brief Expose the contaienr to internal */
    friend class NDArray;
  };

  /*! \brief default constructor */
  NDArray() {}

  /*! \brief default constructor */
  explicit NDArray(std::shared_ptr<Container> data) : data_(data) {}

  /*! \return The dl tensor repr of the array */
  const DLTensor *get() const { return &(data_->dl_tensor); }

  /*! \return The dltensorrepr of the ndarray */
  const DLTensor *operator->() const { return get(); }

  /*! \return the underlying shape tuple. */
  const ShapeTuple &Shape() const { return data_->shape_; }

  /*! \return The dl tensor repr of the array */
  DLTensor *get_mutable() { return &(data_->dl_tensor); }
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   */
  void CopyFrom(const DLTensor *other) {
    CopyFromTo(other, &(data_->dl_tensor));
  }
  void CopyFrom(const NDArray &other) {
    CopyFromTo(other.get(), &(data_->dl_tensor));
  }
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   */
  void CopyTo(DLTensor *other) const { CopyFromTo(&(data_->dl_tensor), other); }
  void CopyTo(const NDArray &other) const {
    CopyFromTo(&(data_->dl_tensor), &(other.data_->dl_tensor));
  }
  /*!
   * \brief Copy data content from a byte buffer.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the buffer in bytes
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a Synchronize
   */
  void CopyFromBytes(const void *data, size_t nbytes);
  /*!
   * \brief Copy data content into another array.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the data buffer.
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a TVMSynchronize.
   */
  void CopyToBytes(void *data, size_t nbytes) const;
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param dev The device of the array.
   * \param mem_scope The memory scope of the array.
   * \return The created array
   */
  static NDArray Empty(ShapeTuple shape, DLDataType dtype, DLDevice dev);
  /*!
   * \return number of elements that the ndarray contains.
   */
  size_t NumElements() const { return GetNumElements(get()); }

  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  static void CopyFromTo(const DLTensor *from, DLTensor *to,
                         DLStreamHandle stream = nullptr);

  /*!
   * Check if a DLTensor is backed by a contiguous memory
   * and byte_offset == 0
   *
   * \param arr The array to be checked.
   * \return the check result.
   *
   * \note To simplify the project, we only handle
   *       compact DLTensor in most part of the codebase.
   */
  static bool IsCompact(const DLTensor *arr);

  /*!
   * \brief Calculate the total number of bytes the DLTensor hold.
   *
   *  \param arr the input DLTensor
   *  \return number of  bytes of data in the DLTensor.
   */
  static size_t GetDataSize(const DLTensor *arr);

  /*!
   * \brief Calculate the total number of elements the DLTensor hold.
   *
   *  \param arr the input DLTensor
   *  \return number of  bytes of data in the DLTensor.
   */
  static size_t GetNumElements(const DLTensor *arr);

private:
  /*! \brief Internal data of the shared ptr */
  std::shared_ptr<Container> data_;
};

/*!
 * \brief convert DLDataType to string repr.
 * \param dtype The data type.
 * \return The string representation.
 */
std::string DLDataType2String(DLDataType t);

/*!
 * \brief convert String to DLDataType.
 * \param s The string representation
 * \return The corresponding DLDataType.
 */
DLDataType String2DLDataType(std::string s);

/*!
 * \return whether the type matches the pattern.
 * \param dtype The data type.
 * \param code The type code.
 * \param bits The bits field.
 * \param lanes The lanes field.
 */
inline bool DLTypeMatch(DLDataType dtype, DLDataTypeCode code, int bits = 32,
                        int lanes = 1) {
  return dtype.code == code && dtype.bits == bits && dtype.lanes == lanes;
}

} // namespace needle
#endif // DLDYS_RUNTIME_NDARRAY_H_
