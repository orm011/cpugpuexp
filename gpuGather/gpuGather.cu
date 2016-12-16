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

const int kWarpSize = 32;

inline int xorshift_hash(int x) {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return ((unsigned int)x) * 213338717U;
}


// aka.1024. worked slightly better.
// it means each of the 4 exec units has 8 threads it can try to schedule
// and can hide latency up to 8x of

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

enum Variant {
  Mat,
  NoMat,
  OnlyMat,
  NaiveMemcpy,
  SingleThread,
  CudaMemcpy,
  CudaMemset,
  MAXVARIANT // do not use.
};

template <Variant variant, int GrainSize> __global__ void
templateKernel(const int * __restrict__ index_col,
               const int *__restrict__ dimension_col,
               int * __restrict__ output,
               int idx_len,
               int idx_domain)
{
  const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  const auto offset = GrainSize*tid;
  const auto unknown_variant = 0;
  const auto mask = idx_domain - 1;
  
  for (int g = 0; g < GrainSize; ++g){
    auto item = offset + g*kWarpSize;
    
    if (item < idx_len)
      {

        switch(variant){
        case NaiveMemcpy:
          output[item] = index_col[item];
          break;
        case Mat:
          output[item] = dimension_col[index_col[item]];
          break;
        case OnlyMat:
          output[item] = 5*index_col[item] + 1;
          break;
        case NoMat:
          {
            auto x = item;
            auto a = x ^ (x >> 12); 
            auto b = a ^ (a << 25);
            auto c = b ^ (b >> 27);
            
            auto d  = ((unsigned int)c) * 213338717U;
            auto idx = d & mask;
            output[item] = dimension_col[idx];
            break;
          }
        default:
          assert(unknown_variant);
        }
      }
  }
}

template <Variant variant, int GrainSize> __global__ void
templateKernelILP(const int * __restrict__ index_col,
               const int *__restrict__ dimension_col,
               int * __restrict__ output,
               int idx_len,
               int idx_domain)
{
  const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  const auto offset = GrainSize*tid;
  const auto unknown_variant = 0;
  const auto mask = idx_domain - 1;
  const int kCacheLine = 32;

  struct cline {
    long int line[kCacheLine/sizeof(long int)];
  };
  
  // most blocks.
  if (((blockIdx.x + 1) * blockDim.x)*GrainSize < idx_len) {
  int tmp[GrainSize];
  cline lines[GrainSize];
    
  // load phase
  for (int g = 0; g < GrainSize; ++g) {
    auto item = offset + g*kWarpSize;
        switch(variant) {
        case SingleThread: {
          if (threadIdx.x == 0){
            lines[g] = *(cline*)(&index_col[item]);
          }
          break;
        }
        case NaiveMemcpy:
          tmp[g] = index_col[item];
          break;
        case Mat:
          tmp[g] = dimension_col[index_col[item]];
          break;
        case OnlyMat:
          tmp[g] = 5*index_col[item] + 1;
          break;
        case NoMat:
          {
            auto x = item;
            auto a = x ^ (x >> 12); 
            auto b = a ^ (a << 25);
            auto c = b ^ (b >> 27);
            
            auto d  = ((unsigned int)c) * 213338717U;
            auto idx = d & mask;
            tmp[g] = dimension_col[idx];
            break;
          }
        default:
          assert(unknown_variant);
        }
  }
    // use phase
    for (int g = 0; g < GrainSize; ++g){
      auto item = offset + g*kWarpSize;
      switch(variant){
      case SingleThread:{
        if (threadIdx.x == 0){
          *((cline*)&output[item]) = lines[g];
        }
        break;
      }
      default:
        output[item] = tmp[g];
      }
    }

  } else {
    for (int g = 0; g < GrainSize; ++g){
    auto item = offset + g*kWarpSize;
    
    if (item < idx_len)
      {

        switch(variant){
        case SingleThread:
        case NaiveMemcpy:
          output[item] = index_col[item];
          break;
        case Mat:
          output[item] = dimension_col[index_col[item]];
          break;
        case OnlyMat:
          output[item] = 5*index_col[item] + 1;
          break;
        case NoMat:
          {
            auto x = item;
            auto a = x ^ (x >> 12); 
            auto b = a ^ (a << 25);
            auto c = b ^ (b >> 27);
            
            auto d  = ((unsigned int)c) * 213338717U;
            auto idx = d & mask;
            output[item] = dimension_col[idx];
            break;
          }
        default:
          assert(unknown_variant);
        }
      }
  }
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


using KernelT = void(const int *, const int *, int *, int, int);
template <Variant variant, int GrainSize, int ThreadsPerBlock>
void GPU_BM(benchmark::State& state)
{
  static_assert(variant < MAXVARIANT, "invalid variant");
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
    
    const int itemsPerBlock = ThreadsPerBlock*GrainSize;
    const int blocksPerGrid = (idx_size + itemsPerBlock - 1) / itemsPerBlock;
    
    cudaCheckErrors(cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice));

    KernelT* kernel = nullptr;
    switch (variant){
    case CudaMemcpy:
    case CudaMemset:
      break;
    default:
      kernel=templateKernelILP<variant, GrainSize>;
    }

    fprintf(stderr,
            "Variant: %d.\n"
            "Grain size: %d.\n"
            "Threads per block: %d.\n"
            "Blocks per SM: %d.\n"
            "Remainder blocks: %d.\n"
            "Remainder threads: %d.\n",
            variant,
            GrainSize,
            ThreadsPerBlock,
            blocksPerGrid / 24,
            blocksPerGrid % 24,
            2048 % ThreadsPerBlock);

    while (state.KeepRunning()){
        switch (variant) {
        case CudaMemcpy:
          cudaMemcpy(d_C, d_A, idx_size, cudaMemcpyDeviceToDevice);
          break;
        case CudaMemset:
          cudaMemset(d_C, 0xf, idx_size);
          break;
        default:
          kernel<<<blocksPerGrid, ThreadsPerBlock>>>(d_A, d_B, d_C, idx_num, dim_num);
          break;
        }
        
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(int64_t(state.iterations())*int64_t(idx_num));

    switch(variant){
    case CudaMemcpy:
    case NaiveMemcpy:
    case SingleThread:
      state.SetBytesProcessed(int64_t(state.iterations())*
                              int64_t(idx_size * 2)); // read write
      break;
    case CudaMemset:
      state.SetBytesProcessed(int64_t(state.iterations())*
                              int64_t(idx_size)); // read write
      break;
    default:
      break;
    }
    
    // Allocate the host output vector C for checking.
    int *h_C = nullptr;
    cudaCheckErrors(cudaMallocHost(&h_C, idx_size));
    cudaCheckErrors(cudaMemcpy(h_C, d_C, idx_size, cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    switch (variant){
    case CudaMemcpy:
    case NaiveMemcpy:
    case SingleThread:
      { 
        for (int i = 0; i < idx_num; ++i){
          if (h_C[i] != h_A[i]) {
            state.SkipWithError("memcpy verification failed");
            break;
          }
        }
        break;
      }
    case CudaMemset:
      {
        for (int i = 0; i < idx_num; ++i){
          if (h_C[i] != 0x0f0f0f0f){
            state.SkipWithError("memset verification failed");
            break;
          }
        }
        break;
      }
    default:
      {
        for (int i = 0; i < idx_num; ++i) {
            if (h_C[i] != h_A[i]*5+1) {
                state.SkipWithError("gather verification failed");
                break;
            }
        }
        break;
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

#define TPB  (4*32*8)
#define GRAIN_1 1

BENCHMARK_TEMPLATE(GPU_BM, Mat, GRAIN_1, TPB)
->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 1 << G}})
->Unit(benchmark::kMillisecond); 

BENCHMARK_TEMPLATE(GPU_BM, NoMat, GRAIN_1, TPB) // actually does write output for now..
->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 1 << G}})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, OnlyMat, GRAIN_1, TPB)  // dim should be irrelevant
->Args({1 << G, 1 <<K})->Args({1<<G, 1<<G})
->Unit(benchmark::kMillisecond); 

BENCHMARK_TEMPLATE(GPU_BM, NaiveMemcpy, 1, TPB) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, NaiveMemcpy, 2, TPB) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);


BENCHMARK_TEMPLATE(GPU_BM, NaiveMemcpy, 4, TPB) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, SingleThread, 1, TPB/2) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, SingleThread, 2, TPB/2) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);


BENCHMARK_TEMPLATE(GPU_BM, SingleThread, 4, TPB/2) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMicrosecond);


BENCHMARK_TEMPLATE(GPU_BM, CudaMemcpy, GRAIN_1, TPB) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(GPU_BM, CudaMemset, GRAIN_1, TPB) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMicrosecond);


BENCHMARK_MAIN();