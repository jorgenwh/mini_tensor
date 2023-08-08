#pragma once

#include <stdio.h>
#include <cudnn.h>

#define CUDNN_CHECK(status) { cudnn_errcheck(status, __FILE__, __LINE__); }

inline void cudnn_errcheck(cudnnStatus_t status, const char* file, int line) {
#ifdef __CUDNN_ERROR_CHECK__
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("cuDNN error: %s, %s, %d\n", cudnnGetErrorString(status), file, line);
    exit(EXIT_FAILURE);
  }
#endif // __CUDNN_ERROR_CHECK__
}
