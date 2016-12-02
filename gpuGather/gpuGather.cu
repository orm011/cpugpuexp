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
#include <benchmark/benchmark.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <helper_cuda.h>
#include <cassert>
#include <iostream>

using namespace std;

// TODO: enable this in order to try mapped memory (vs streaming)
// cudaSetDeviceFlags(cudaDeviceMapHost);
// Print the vector length to be used, and compute its size
const int G = 30;
//const int M = 20;
const int K = 10;

inline int xorshift_hash(int x) {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return ((unsigned int)x) * 213338717U;
}

enum DoVerify {
  VERIFY,
  NO_VERIFY
};

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
vectorGather(const int * __restrict__ index_col, const int *__restrict__ dimension_col, int * __restrict__ output, int idx_len, int idx_domain)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < idx_len)
    {
      output[i] = dimension_col[index_col[i]];
    }
}


// used as control
__global__ void
vectorCopy(const int * __restrict__ index_col, const int *__restrict__, int * __restrict__ output, int idx_len, int idx_domain)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < idx_len)
    {
      output[i] = index_col[i];
    }
}


__global__ void
vectorGatherNoInput(const int *__restrict__ , const int * __restrict__ dimension_col, int * __restrict__ output, int idx_len, int idx_domain)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    const auto mask = idx_domain - 1;
    
    if (i < idx_len)
    {
      auto x = i;
      auto a = x ^ (x >> 12); 
      auto b = a ^ (a << 25);
      auto c = b ^ (b >> 27);
      
      auto d  = ((unsigned int)c) * 213338717U;
      auto idx = d & mask;
      output[i] = dimension_col[idx];
    }
}

// basically CPU + write, to see how much is probably random.
__global__ void
vectorCopyNoInput(const int *__restrict__ dimension_col, int * __restrict__ output, int idx_len, int mask)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < idx_len)
    {
      auto x = i;
      auto a = x ^ (x >> 12); 
      auto b = a ^ (a << 25);
      auto c = b ^ (b >> 27);
      
      auto d  = ((unsigned int)c) * 213338717U;
      auto idx = d & mask;
      output[i] = idx;
    }
}


#define cudaCheckErrors($call)                     \
    do { \
      cudaError_t err = cudaGetLastError(); \
      if (err != cudaSuccess){\
        fprintf(stderr, "WARNING: Error was already set before call: (%s at %s:%d)\n", \
                cudaGetErrorString(err),                       \
                __FILE__, __LINE__); \
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


//using kernel_t = void(const int *, const int *, int *, int, int);
template <typename Kernel>
void benchmark_base(benchmark::State& state, Kernel kernel, DoVerify verify_correct)
{
  //cerr << "running bench again" << endl;
  //printf("FYI:\ncuda cpuDeviceId: %d\n", cudaCpuDeviceId);
  int64_t idx_size = state.range(0);
  int64_t dim_size = state.range(1);
  int64_t idx_num = idx_size / sizeof(int);
  int64_t dim_num = dim_size / sizeof(int);

  //auto dev = 0;
  //cudaCheckErrors(cudaSetDevice(dev));
  //cudaDeviceProp deviceProp;
  //cudaCheckErrors(cudaGetDeviceProperties(&deviceProp, dev));

  // printf("some device %d properties:\n",dev);
  // printf("concurrent kernels %d\n",deviceProp.concurrentKernels);
  // printf("device overlap %d\n",deviceProp.deviceOverlap);
  // printf("max threads per block %d\n",deviceProp.maxThreadsPerBlock);
  // printf("warp size %d\n",deviceProp.warpSize);
  // printf("regs per block %d\n",deviceProp.regsPerBlock);
  // printf("[Gather of %lu indices into a table of %lu locations]\n", idx_num, dim_num);

    // Allocate the host input vector A
    int *h_A = nullptr;
    cudaCheckErrors(cudaMallocHost(&h_A, idx_size));

    // Allocate the host input vector B
    int *h_B = nullptr;
    cudaCheckErrors(cudaMallocHost(&h_B, dim_size));
    
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
    cudaCheckErrors(cudaMalloc(&d_A, idx_size));

    // Allocate the device input vector B
    int *d_B = NULL;
    cudaCheckErrors(cudaMalloc((void **)&d_B, dim_size));

    // Allocate the device output vector C
    int *d_C = NULL;
    cudaCheckErrors(cudaMalloc((void **)&d_C, idx_size));

    const int threadsPerBlock = 256; // tried tuning. 
    const int blocksPerGrid = (idx_size + threadsPerBlock - 1) / threadsPerBlock;
    // cudaStream_t streams[kNumStreams];
    // for (int i = 0; i < kNumStreams; ++i) {
    //   cudaStreamCreate(&streams[i]);
    // }
    // auto part_num = idx_num/kNumStreams;
    // auto part_sz = idx_size/kNumStreams;
    
    // Copy the host input vectors A and B in host memory to the device input vectors in device memory
    // printf("Copy idx from the host memory to the CUDA device\n");
    // printf("CUDA kernel launch with %d blocks of %d threads over %d streams\n", blocksPerGrid, threadsPerBlock, kNumStreams);
    cudaCheckErrors(cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice));

    while (state.KeepRunning())
    {
      kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, idx_num, dim_num);
      cudaDeviceSynchronize();
    }

    //vectorCopyNoInput<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, idx_num, mask);
    //vectorGatherNoInput<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, idx_num, mask);

    // Allocate the host output vector C for checking.
    int *h_C = nullptr;
    cudaCheckErrors(cudaMallocHost(&h_C, idx_size));
    cudaCheckErrors(cudaMemcpy(h_C, d_C, idx_size, cudaMemcpyDeviceToHost));
    
    // Verify that the result vector is correct
    if (verify_correct == VERIFY){
      for (int i = 0; i < idx_num; ++i)
        {
          if (h_C[i] != h_A[i]*5+1)
            {
              fprintf(stdout, "Result verification first failed at element %d!\n", i);
              break;
              //exit(0);
              //exit(EXIT_FAILURE);
            }
        }
    }

    cudaCheckErrors(cudaFreeHost(h_A));
    cudaCheckErrors(cudaFreeHost(h_B));
    cudaCheckErrors(cudaFreeHost(h_C));
    cudaCheckErrors(cudaFree(d_A));
    cudaCheckErrors(cudaFree(d_B));
    cudaCheckErrors(cudaFree(d_C));
    //printf("Test PASSED.\n");
}



BENCHMARK_CAPTURE(benchmark_base, vectorGather, vectorGather, VERIFY)
->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 1 << G}})
->Unit(benchmark::kMillisecond); 

BENCHMARK_CAPTURE(benchmark_base, vectorCopy, vectorCopy, NO_VERIFY) // the dim table is irrelevant here
->Args({1 << G, 4}) 
->Unit(benchmark::kMillisecond); 

BENCHMARK_MAIN();