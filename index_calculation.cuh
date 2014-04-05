#pragma once

#include <cuda.h>

__device__ inline unsigned long long int getGlobalBlockId() {
   return blockIdx.x +
          blockIdx.y * gridDim.x +
          blockIdx.z * gridDim.x * gridDim.y;
}

__device__ unsigned long long int getBlockThreadId() {
   return threadIdx.x + 
          threadIdx.y * blockDim.x +
          threadIdx.z * blockDim.x * blockDim.y;
}

__device__ inline unsigned long long int getGlobalThreadId() {
   return getGlobalBlockId() * 
          blockDim.x * blockDim.y * blockDim.z +
          getBlockThreadId();
}

__device__ inline int getTotalThreadsCount() { 
   return blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
}

