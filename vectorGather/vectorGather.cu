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
      $call;                                  \
      cudaError_t __err = cudaGetLastError(); \
      if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: (%s at %s:%d)\n", \
                cudaGetErrorString(__err), \
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
  auto dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaCheckErrors(cudaGetDeviceProperties(&deviceProp, dev));
  printf("concurrent kernels %d\n",deviceProp.concurrentKernels);
  printf("device overlap %d\n",deviceProp.deviceOverlap);
  printf("max threads per block %d\n",deviceProp.maxThreadsPerBlock);
  printf("max threads per block %d\n",deviceProp.maxThreadsPerBlock);
  
  // TODO: enable this in order to try mapped memory (vs streaming)
  // cudaSetDeviceFlags(cudaDeviceMapHost);
  
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

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
    // this call fails for with invalid device for some unknown reason...
    // cudaCheckErrors(cudaMemAdvise(h_C, idx_size, cudaMemAdviseSetPreferredLocation, dev));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

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

    cudaMemAdvise(h_A, idx_size, cudaMemAdviseSetReadMostly, 0);
    cudaCheckErrors();
    cudaMemAdvise(h_B, dim_size, cudaMemAdviseSetReadMostly, 0);
    cudaCheckErrors();

    // // Allocate the device input vector A
    // int *d_A = NULL;
    // err = cudaMalloc((void **)&d_A, idx_size);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // Allocate the device input vector B
    // int *d_B = NULL;
    // err = cudaMalloc((void **)&d_B, dim_size);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // // Allocate the device output vector C
    // // int *d_C = NULL;
    // // err = cudaMalloc((void **)&d_C, idx_size);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // Copy the host input vectors A and B in host memory to the device input vectors in
    // // device memory
    // printf("Copy idx from the host memory to the CUDA device\n");
    // err = cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // } 

    using namespace std::chrono;

    
    // err = cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    printf("FYI, cuda cpuDeviceId: %d\n", cudaCpuDeviceId);
    const int threadsPerBlock = 256; // try tuning this... no?
    const int blocksPerGrid = (idx_size + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch the Vector Add CUDA Kernel
    auto start = high_resolution_clock::now();
    
    cudaMemPrefetchAsync(h_B, dim_size, dev);     
    cudaMemPrefetchAsync(h_A, idx_size, dev);     
    vectorGather<<<blocksPerGrid, threadsPerBlock>>>(h_A, h_B, h_C, idx_num);
    cudaMemPrefetchAsync(h_C, idx_size, cudaCpuDeviceId);
    err = cudaDeviceSynchronize();

    auto end   = high_resolution_clock::now();
    auto diff = duration_cast<milliseconds>(end - start).count();

    printf("kernel runtime: %ld ms\n", diff);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
        
    // // Copy the device result vector in device memory to the host result vector
    // // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    // err = cudaMemcpy(h_C, d_C, idx_size, cudaMemcpyDeviceToHost);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Verify that the result vector is correct
    for (int i = 0; i < idx_num; ++i)
    {
      if (h_C[i] != h_A[i]*5+1)
        {
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED.");

    // // Free device global memory
    // err = cudaFree(d_A);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaFree(d_B);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaFree(d_C);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Free host memory
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    printf("Done\n");
    return 0;
}

