#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(status) { cuda_errcheck(status, __FILE__, __LINE__); }

inline void cuda_errcheck(cudaError_t status, const char* file, int line) {
#ifdef __CUDA_ERROR_CHECK__
  if (status != cudaSuccess) {
    printf("CUDA error: %s, %s, %d\n", cudaGetErrorString(status), file, line);
    exit(EXIT_FAILURE);
  }
#endif // __CUDA_ERROR_CHECK__
}
