#include <needle/cuda_ops.h>
#include <needle/logging.h>
#include <needle/ndarray.h>

namespace needle {
namespace cuda {

const int kBaseThreadNum = 256;

// namespace for op functors
namespace op {

struct Add {
  static __device__ float Run(float lhs, float rhs) { return lhs + rhs; }
};

struct Mul {
  static __device__ float Run(float lhs, float rhs) { return lhs * rhs; }
};
} // namespace op

// whether the array is compact f32 array
inline bool IsCompactF32(const DLTensor *arr) {
  return NDArray::IsCompact(arr) && DLTypeMatch(arr->dtype, kDLFloat, 32);
}

__global__ void FillKernel(float value, float *output, int num_elements) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < num_elements) {
    output[gid] = value;
  }
}

void Fill(float value, NDArray out) {
  int num_elements = NDArray::GetNumElements(out.get());
  CHECK(IsCompactF32(out.get()));

  int num_blocks = (num_elements + kBaseThreadNum - 1) / kBaseThreadNum;

  dim3 dimBlock(kBaseThreadNum, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);
  // launch kernel
  FillKernel<<<dimGrid, dimBlock>>>(value, static_cast<float *>(out->data),
                                    num_elements);
}

template <typename Op>
__global__ void EWiseBinaryKernel(const float *lhs, const float *rhs,
                                  float *output, int num_elements) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < num_elements) {
    output[gid] = Op::Run(lhs[gid], rhs[gid]);
  }
}

template <typename Op>
void EWiseOp(const DLTensor *lhs, const DLTensor *rhs, DLTensor *out) {
  int num_elements = NDArray::GetNumElements(lhs);
  CHECK(IsCompactF32(lhs));
  CHECK(IsCompactF32(rhs));
  CHECK(IsCompactF32(out));

  int num_blocks = (num_elements + kBaseThreadNum - 1) / kBaseThreadNum;

  dim3 dimBlock(kBaseThreadNum, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);
  // launch kernel
  EWiseBinaryKernel<Op><<<dimGrid, dimBlock>>>(
      static_cast<float *>(lhs->data), static_cast<float *>(rhs->data),
      static_cast<float *>(out->data), num_elements);
}

void EWiseAdd(NDArray lhs, NDArray rhs, NDArray out) {
  EWiseOp<op::Add>(lhs.get(), rhs.get(), out.get_mutable());
}

void EWiseMul(NDArray lhs, NDArray rhs, NDArray out) {
  EWiseOp<op::Mul>(lhs.get(), rhs.get(), out.get_mutable());
}

template <typename Op>
__global__ void BinaryScalarKernel(const float *lhs, float scalar,
                                   float *output, int num_elements) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < num_elements) {
    output[gid] = Op::Run(lhs[gid], scalar);
  }
}

template <typename Op>
void BinaryScalarOp(const DLTensor *lhs, float scalar, DLTensor *out) {
  int num_elements = NDArray::GetNumElements(lhs);
  // specialize to float32 compact array
  CHECK(IsCompactF32(lhs));
  CHECK(IsCompactF32(out));

  int num_blocks = (num_elements + kBaseThreadNum - 1) / kBaseThreadNum;

  dim3 dimBlock(kBaseThreadNum, 1, 1);
  dim3 dimGrid(num_blocks, 1, 1);
  // launch kernel
  BinaryScalarKernel<Op>
      <<<dimGrid, dimBlock>>>(static_cast<float *>(lhs->data), scalar,
                              static_cast<float *>(out->data), num_elements);
}

void AddScalar(NDArray lhs, float scalar, NDArray out) {
  BinaryScalarOp<op::Add>(lhs.get(), scalar, out.get_mutable());
}

void MulScalar(NDArray lhs, float scalar, NDArray out) {
  BinaryScalarOp<op::Mul>(lhs.get(), scalar, out.get_mutable());
}

} // namespace cuda
} // namespace needle
