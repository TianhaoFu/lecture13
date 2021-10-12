/*!
 * \file needle/cuda_ops.h
 */
#ifndef NEEDLE_CUDA_OPS_H_
#define NEEDLE_CUDA_OPS_H_

#include <memory>
#include <needle/device_api.h>
#include <needle/ndarray.h>
#include <vector>

namespace needle {
namespace cuda {

void Fill(float value, NDArray out);

void EWiseAdd(NDArray lhs, NDArray rhs, NDArray out);

void AddScalar(NDArray lhs, float scalar, NDArray out);

void EWiseMul(NDArray lhs, NDArray rhs, NDArray out);

void MulScalar(NDArray lhs, float scalar, NDArray out);

} // namespace cuda
} // namespace needle

#endif // NEEDLE_CUDA_OPS_H_
