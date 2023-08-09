#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(status) { cuda_errcheck(status, __FILE__, __LINE__); }
#define CUDNN_CHECK(status) { cudnn_errcheck(status, __FILE__, __LINE__); }

inline void cuda_errcheck(cudaError_t status, const char* file, int line) {
#ifdef __CUDA_ERROR_CHECK__
  if (status != cudaSuccess) {
    printf("CUDA error: %s, %s, %d\n", cudaGetErrorString(status), file, line);
    exit(EXIT_FAILURE);
  }
#endif // __CUDA_ERROR_CHECK__
}

inline void cudnn_errcheck(cudnnStatus_t status, const char* file, int line) {
#ifdef __CUDNN_ERROR_CHECK__
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("cuDNN error: %s, %s, %d\n", cudnnGetErrorString(status), file, line);
    exit(EXIT_FAILURE);
  }
#endif // __CUDNN_ERROR_CHECK__
}
