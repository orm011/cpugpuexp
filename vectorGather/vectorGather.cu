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


/**
 * Host main routine
 */
int
main(void)
{
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
    int *h_A = (int *)malloc(idx_size);

    // Allocate the host input vector B
    int *h_B = (int *)malloc(dim_size);

    // Allocate the host output vector C
    int *h_C = (int *)malloc(idx_size);

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

    // Allocate the device input vector A
    int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, idx_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, dim_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, idx_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    auto time_it = [](std::function<void()> fn){
      auto start = std::chrono::high_resolution_clock::now();
      fn();
      auto end   = std::chrono::high_resolution_clock::now();
      auto diff  =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      return (uint32_t)diff.count();
    };

    
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    auto copy_dim = [&](){
      err = cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice);
    };
    
    auto copy_dim_millis = time_it(copy_dim);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(idx_size + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


    auto kernel_exec = [&](){
      vectorGather<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, idx_num);
      err = cudaDeviceSynchronize();
    };
    
    uint32_t kernel_millis = time_it(kernel_exec);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
        
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    auto copy_back = [&](){
      err = cudaMemcpy(h_C, d_C, idx_size, cudaMemcpyDeviceToHost);
    };

    auto copy_millis = time_it(copy_back);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < idx_num; ++i)
    {
      if (h_C[i] != h_A[i]*5+1)
        {
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED. copy dim (ms): %u. kernel time (ms): %u. copy back time (ms): %u\n",
           copy_dim_millis, kernel_millis, copy_millis);

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
