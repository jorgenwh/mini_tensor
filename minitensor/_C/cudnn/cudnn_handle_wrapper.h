#pragma once

#include <cudnn.h>
#include "err.h"

class CuDNNHandleWrapper {
public:
  CuDNNHandleWrapper() {
    CUDNN_CHECK(cudnnCreate(&handle_));
  };
  ~CuDNNHandleWrapper() {
    if (handle_ != nullptr) {
      CUDNN_CHECK(cudnnDestroy(handle_));
    }
  }
  cudnnHandle_t get() const {
    return handle_;
  }
private:
  cudnnHandle_t handle_ = nullptr;
};
