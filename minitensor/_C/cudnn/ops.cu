#include <cuda_runtime.h>
#include <cudnn.h>

#include "err.h"
#include "ops.h"

namespace ops {

float sum(const float *in, const int size, cudnnHandle_t cudnn_handle) {
  cudnnTensorDescriptor_t in_desc, out_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, size));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));

  cudnnReduceTensorDescriptor_t reduce_desc;
  CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc));
  CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc, 
        CUDNN_REDUCE_TENSOR_ADD, 
        CUDNN_DATA_FLOAT, 
        CUDNN_NOT_PROPAGATE_NAN, 
        CUDNN_REDUCE_TENSOR_NO_INDICES, 
        CUDNN_32BIT_INDICES));

  size_t workspace_size;
  CUDNN_CHECK(cudnnGetReductionWorkspaceSize(cudnn_handle, reduce_desc, in_desc, out_desc, &workspace_size));
  size_t indices_size;
  CUDNN_CHECK(cudnnGetReductionIndicesSize(cudnn_handle, reduce_desc, in_desc, out_desc, &indices_size));
  void *workspace_buffer;
  CUDA_CHECK(cudaMalloc(&workspace_buffer, workspace_size));
  void *indices_buffer;
  CUDA_CHECK(cudaMalloc(&indices_buffer, indices_size));

  float alpha = 1.0f;
  float beta = 0.0f;

  // reduce
  float *result;
  cudaMalloc(&result, sizeof(float));
  CUDNN_CHECK(cudnnReduceTensor(
        cudnn_handle,
        reduce_desc,
        indices_buffer,
        indices_size,
        workspace_buffer,
        workspace_size,
        &alpha,
        in_desc,
        in,
        &beta,
        out_desc,
        result));

  float ret;
  CUDA_CHECK(cudaMemcpy(&ret, result, sizeof(float), cudaMemcpyDeviceToHost));

  // clean-up
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduce_desc));
  CUDA_CHECK(cudaFree(workspace_buffer));
  CUDA_CHECK(cudaFree(indices_buffer));
  CUDA_CHECK(cudaFree(result));

  return ret;
}

} // namespace ops
