/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <helper_cuda.h>
#include <cassert>

inline int xorshift_hash(int x) {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return ((unsigned int)x) * 213338717U;
}


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
// __global__ void
// vectorAdd(const float *A, const float *B, float *C, int numElements)
// {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;

//     if (i < numElements)
//     {
//         C[i] = A[i] + B[i];
//     }
// }

__global__ void
vectorGather(const int *index_col, const int *dimension_col, int *output, int idx_len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < idx_len)
    {
      output[i] = dimension_col[index_col[i]];
    }
}


__global__ void
vectorGatherNoScan(const int *dimension_col, int *output, int idx_len, int mask)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < idx_len)
    {
      auto x = i;
      x ^= x >> 12; // a
      x ^= x << 25; // b
      x ^= x >> 27; // c
      x = ((unsigned int)x) * 213338717U;
      auto idx = x & mask;
      output[i] = dimension_col[idx];
    }
}

#define cudaCheckErrors($call)                     \
    do { \
      cudaError_t err = cudaGetLastError(); \
      if (err != cudaSuccess){\
        fprintf(stderr, "Error already set before call: (%s at %s:%d)\n", \
                cudaGetErrorString(err),                       \
                __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
      }\
      $call;                                  \
      err = cudaGetLastError(); \
      if (err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: (%s at %s:%d)\n", \
                cudaGetErrorString(err),                \
                __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
      } \
    } while (0)


/**
 * Host main routine
 */
int
main(void)
{
  printf("FYI:\ncuda cpuDeviceId: %d\n", cudaCpuDeviceId);
  auto dev = 0;
  cudaCheckErrors(cudaSetDevice(dev));
  cudaDeviceProp deviceProp;
  cudaCheckErrors(cudaGetDeviceProperties(&deviceProp, dev));

  printf("some device %d properties:\n",dev);
  printf("concurrent kernels %d\n",deviceProp.concurrentKernels);
  printf("device overlap %d\n",deviceProp.deviceOverlap);
  printf("max threads per block %d\n",deviceProp.maxThreadsPerBlock);
  printf("warp size %d\n",deviceProp.warpSize);
  printf("regs per block %d\n",deviceProp.regsPerBlock);
  
  // TODO: enable this in order to try mapped memory (vs streaming)
  // cudaSetDeviceFlags(cudaDeviceMapHost);
  
    // Print the vector length to be used, and compute its size
    const int G = 30;
    const int M = 20;
    //const int K = 10;
    
    size_t idx_size = 1U << G;
    size_t dim_size = 256U << M;
    
    size_t idx_num = idx_size / sizeof(int);
    size_t dim_num = dim_size / sizeof(int);
    
    printf("[Gather of %lu indices into a table of %lu locations]\n", idx_num, dim_num);

    // Allocate the host input vector A
    int *h_A = nullptr;
    cudaCheckErrors(cudaMallocManaged(&h_A, idx_size));

    // Allocate the host input vector B
    int *h_B = nullptr;
    cudaMallocManaged(&h_B, dim_size);
    cudaCheckErrors();
    
    // Allocate the host output vector C
    int *h_C = nullptr;
    cudaCheckErrors(cudaMallocManaged(&h_C, idx_size));

    int sm = __builtin_popcountl (dim_num);
    assert(sm == 1); // popcount of 1.
    const int mask = dim_num - 1;

    // Initialize the host input vectors
    for (int i = 0; i < idx_num; ++i)
    {
       h_A[i] = xorshift_hash(i) & mask;
       assert(h_A[i] < dim_num);
    }

    for (int i = 0; i < dim_num; ++i){
      h_B[i] = 5*i + 1;
    }


    // Allocate the device input vector A
    int *d_A = NULL;
    cudaCheckErrors(cudaMalloc((void **)&d_A, idx_size));

    // Allocate the device input vector B
    int *d_B = NULL;
    cudaCheckErrors(cudaMalloc((void **)&d_B, dim_size));

    // Allocate the device output vector C
    int *d_C = NULL;
    cudaCheckErrors(cudaMalloc((void **)&d_C, idx_size));


    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy idx from the host memory to the CUDA device\n");
    cudaCheckErrors(cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice));

    using namespace std::chrono;
    const int threadsPerBlock = 256; // try tuning this... no?
    const int blocksPerGrid = (idx_size + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch the Vector Add CUDA Kernel
    auto start = high_resolution_clock::now();    
    vectorGather<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, idx_num);
    cudaCheckErrors(cudaDeviceSynchronize());
    auto end   = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds>(end - start).count();
    cudaCheckErrors();
    
    cudaCheckErrors(cudaMemcpy(h_C, d_C, idx_size, cudaMemcpyDeviceToHost));
    printf("kernel runtime: %ld ms\n", diff);
    
    // Verify that the result vector is correct
    for (int i = 0; i < idx_num; ++i)
    {
      if (h_C[i] != h_A[i]*5+1)
        {
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED.\n");
    return 0;
}

