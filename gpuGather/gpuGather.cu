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
constexpr int G = 30;
constexpr int M = 20;
constexpr int K = 10;

constexpr int kWarpSize = 32;
constexpr int kNumSM = 24; // gpu specific.

__host__ __device__ inline int xorshift_hash(int x) {
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

enum struct Variant {
  Mat,
  NoMat,
  OnlyMat,
  NaiveMemcpy,
  StreamMemcpy,
  RandCheck, // if it has different performance to Mat, then the RNG is not good.
  CudaMemcpy,
  CudaMemset,
  MAXVARIANT // do not use.
};

// how many outstanding requests (concurrent) in ad.
enum ILP {
  ilp1 = 1,
  ilp2 = 2,
  ilp4 = 4,
  ilp8 = 8,
  ilp16 = 16,
  ilp32 = 32,
  ilp64 = 64,
  ilp128 = 128,
};

enum ActiveLanes {
  al1 = 1,
  al2 = 2,
  al4 = 4,
  al8 = 8,
  al16 = 16,
  al32 = 32
};

// how many tasks of size ILP will be executed in series.
enum GrainSize {
  gs1 = 1,
  gs2 = 2,
  gs3 = 3,
  gs4 = 4,
  gs8 = 8
};

// returns a 1-hot of the current lane.
__device__ inline int get_lane(){
  return threadIdx.x % 32;
  // int lanebit = 0;
  // asm("mov.u32 %0, %lanemask_eq;" : "=r"(lanebit));
  // const auto lane =  __ffs(lanebit) - 1;
  // return lane;
}

template <Variant variant, ILP ilp, ActiveLanes ActiveThreads, GrainSize gs>
__global__ void
templateKernel(const int * __restrict__ index_col,
               const int *__restrict__ dimension_col,
               int * __restrict__ output,
               int idx_len,
               int idx_domain)
{
  static_assert(ActiveThreads <= kWarpSize, "limit");
  static_assert(ActiveThreads > 0, "limit");
  static_assert((ActiveThreads - 1 & ActiveThreads) == 0, "power of 2"); // power of 2
  
  // mapping block to data
  constexpr int64_t warpFraction = kWarpSize / ActiveThreads;
  int64_t blockSize = (gs * ilp * blockDim.x)/warpFraction;
  int64_t blockStart = blockSize * blockIdx.x;
  int64_t blockEnd = blockStart + blockSize;

  // mapping warp to data
  constexpr int64_t dataPerWarp = gs * ilp * ActiveThreads;
  int64_t warpNo = threadIdx.x / kWarpSize;
  int64_t warpOffset = blockStart + warpNo * dataPerWarp;

  const auto unknown_variant = 0;
  const auto mask = idx_domain - 1;
  auto lane = get_lane();
  
  if (lane < ActiveThreads) {
    const auto taskoffset = warpOffset + lane;
  
    // // most blocks.
    if (blockEnd <= idx_len) {
    //   if (variant == Variant::NaiveMemcpy){
    //   // init tmp.
    //   for (int g = 0; g < ilp; ++g){
    //     tmp[g][0] = index_col[taskoffset + g*ActiveThreads];
    //     tmp[g][1] = index_col[taskoffset + g*ActiveThreads + delta];
    //   }
    //   }
    int tmp[ilp]; // wait until next one.
    constexpr auto delta = ilp*ActiveThreads;

    
    for (int iter = 0; iter < gs; ++iter){
      auto offset = taskoffset + iter*delta;
      //auto this_slot = (iter % 3);
      //auto next_slot = (iter + 2) % 3;      
      // load phase

      for (int g = 0; g < ilp; ++g) {
        auto item = offset + g*ActiveThreads;

    
        switch(variant) {
        case Variant::NaiveMemcpy:{
          // aka index_col[item + delta];
          int ldd;
          auto nextaddr = &index_col[item + 2*delta];
          asm("ld.global.cs.u32 %0, [%1];" : "=r"(ldd) : "l"(nextaddr));
          tmp[g] = ldd; 
          //auto nextaddr = &index_col[item + delta];
          // prefetch next round
          //asm("prefetch.global.L2 [%0];" :: "l"(nextaddr));
          break;
        }
        case Variant::RandCheck:
        case Variant::Mat:{
          auto idx = index_col[item];
          tmp[g] = idx;
          //auto theaddr = &dimension_col[idx];
          //asm("prefetch.local.L1 [%0];" :: "l"(theaddr));
          break;
        }
        case Variant::OnlyMat:
          tmp[g] = 5*index_col[item] + 1;
          break;
        case Variant::NoMat:
          {
            auto num = xorshift_hash(item);
            auto theidx = num & mask;
            tmp[g] = theidx;
            //asm("prefetchu.L1 [%0];" :: "l"(theaddr));

            //assert(index_col[item] == idx);
            break;
          }
        default:
          assert(unknown_variant);
        }
      }

      for (int g = 0; g < ilp; ++g){
        switch(variant){
        case Variant::NoMat:
        case Variant::RandCheck:
        case Variant::Mat:
          int val;
          auto addr  = &dimension_col[tmp[g]];
          asm("ld.global.cs.s32 %0, [%1];" : "=r"(val) : "l"(addr));
          tmp[g] = val;
          break;      
        }
      }
  
      // use phase
      for (int g = 0; g < ilp; ++g) {
        auto item = offset + g*ActiveThreads;

        switch(variant){
        case Variant::NoMat:
        case Variant::RandCheck:
        case Variant::Mat:
          output[item] = tmp[g];
          //auto outaddr = &output[item];
          //asm("st.global.cs.s32 [%0], %1;": "=l"(outaddr) , "r"(tmp[g][1]));
          break;
        default:
          output[item] = tmp[g];
        }
      }

    }

  } else { // used only for the last thread block.
    //assert(0);
    for (int iter = 0; iter < gs; ++iter){
      auto offset = taskoffset + iter*ilp*ActiveThreads;
      for (int g = 0; g < ilp; ++g) {
        auto item = offset + g*ActiveThreads;
    
        if (item < idx_len)
          {

            switch(variant){
            case Variant::NaiveMemcpy:
              output[item] = index_col[item];
              break;
            case Variant::RandCheck:
            case Variant::Mat:
              output[item] = dimension_col[index_col[item]];
              break;
            case Variant::OnlyMat:
              output[item] = 5*index_col[item] + 1;
              break;
            case Variant::NoMat:
              {
                auto num = xorshift_hash(item);
                auto idx = num & mask;
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
template <Variant variant, ILP ilp, int ThreadsPerBlock, ActiveLanes ActiveThreads, GrainSize gs>
void GPU_BM(benchmark::State& state)
{
  static_assert(int32_t(variant) < int32_t(Variant::MAXVARIANT), "invalid variant");
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
      int rando = 0;
      if (variant == Variant::RandCheck) {
        auto rando = rand();
      } else {
        rando = xorshift_hash(i);
      }
      
      h_A[i] = rando & mask;
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
    // init memory in device. to detect.
    cudaCheckErrors(cudaMemset(d_C, 0xff, idx_size));


    int itemsPerBlock = -1;
    int blocksPerGrid = -1;
    
    cudaCheckErrors(cudaMemcpy(d_B, h_B, dim_size, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_A, h_A, idx_size, cudaMemcpyHostToDevice));

    KernelT* kernel = nullptr;
    switch (variant){
    case Variant::CudaMemcpy:
    case Variant::CudaMemset:
      break;
    default:{
      kernel=templateKernel<variant, ilp, ActiveThreads, gs>;
      auto threadMultiplier = kWarpSize/ActiveThreads;
      itemsPerBlock = (ilp * gs * ThreadsPerBlock)/threadMultiplier;
      blocksPerGrid = (idx_size + itemsPerBlock - 1) / itemsPerBlock;
      fprintf(stderr,
              "Variant: %d\n"
              "ILP: %d\n"
              "Items per thread: %d\n"
              "Items per block: %d\n"
              "Active threads per warp: %d\n"
              "Threads per block: %d\n"
              "Blocks per SM: %d\n"
              "Remainder blocks: %d\n"
              "Remainder threads: %d\n",
              int(variant),
              ilp,
              gs * ilp,
              itemsPerBlock,
              ActiveThreads,
              ThreadsPerBlock,
              blocksPerGrid / kNumSM,
              blocksPerGrid % kNumSM,
              2048 % ThreadsPerBlock);

    }
    }
    

    

    while (state.KeepRunning()){
        switch (variant) {
        case Variant::CudaMemcpy:
          cudaMemcpy(d_C, d_A, idx_size, cudaMemcpyDeviceToDevice);
          break;
        case Variant::CudaMemset:
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
    case Variant::CudaMemcpy:
    case Variant::NaiveMemcpy:
      state.SetBytesProcessed(int64_t(state.iterations())*
                              int64_t(idx_size * 2)); // read write
      break;
    case Variant::CudaMemset:
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
    case Variant::CudaMemcpy:
    case Variant::NaiveMemcpy:
      { 
        for (int i = 0; i < idx_num; ++i) {
          if (h_C[i] != h_A[i]) {
            fprintf(stdout, "\033[1;31mERROR:\033[0m memcpy verification failed at position %d: h_C=%d but h_A=%d\n", i, h_C[i], h_A[i]);
            break; // free memory
          }
        }
        break;
      }
    case Variant::CudaMemset:
      {
        for (int i = 0; i < idx_num; ++i){
          if (h_C[i] != 0x0f0f0f0f){
            fprintf(stdout,  "ERROR. memset verification failed\n");
            break; // free memory
          }
        }
        break;
      }
    default:
      {// mbold red text
        for (int i = 0; i < idx_num; ++i) {
            if (h_C[i] != h_A[i]*5+1) {
              fprintf(stdout, "\033[1;31mERROR:\033[0m gather verification failed at position %d: h_C=%d but h_A=%d and hA*5 + 1 = %d\n", i, h_C[i], h_A[i], h_A[i]*5+ 1);
              break; // free memory
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

#define TPB(n) n
#define ATh(n) ActiveLanes::al##n
#define ILP(n) (ILP::ilp##n)
#define GS(n) GrainSize::gs##n

BENCHMARK_TEMPLATE(GPU_BM, Variant::NoMat, ILP(2), TPB(256), ATh(32), GS(8)) // actually does write output for now..
->RangeMultiplier(8)
->Ranges({{1<<G, 1<<G}, {64 << M,  64 << M}})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(4), TPB(256), ATh(32), GS(4)) // actually does write output for now..
->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {64 << M, 64 << M}})
->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NoMat, ILP(8), TPB(256), ATh(16)) // actually does write output for now..
// ->RangeMultiplier(2)
// ->Ranges({{1<<G, 1<<G}, {256 << M, 256 << M}})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NoMat, ILP(16), TPB(256), ATh(8)) // actually does write output for now..
// ->RangeMultiplier(2)
// ->Ranges({{1<<G, 1<<G}, {256 << M, 256 << M}})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NoMat, ILP(16), TPB(256), ATh(4)) // actually does write output for now..
// ->RangeMultiplier(2)
// ->Ranges({{1<<G, 1<<G}, {256 << M, 256 << M}})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NoMat, ILP(16), TPB(256), ATh(2)) // actually does write output for now..
// ->RangeMultiplier(2)
// ->Ranges({{1<<G, 1<<G}, {256 << M, 256 << M}})
// ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, Variant::Mat, ILP(2), TPB(256), ATh(32), GS(2)) // actually does write output for now..
->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 512 << M}})
->Unit(benchmark::kMillisecond);


// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(1), TPB(1024), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(2), TPB(1024), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(4), TPB(1024), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(8), TPB(1024), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(16), TPB(1024), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);


// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(8), TPB(512), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(16), TPB(512), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

// BENCHMARK_TEMPLATE(GPU_BM, Variant::NaiveMemcpy, ILP(16), TPB(512), ATh(32)) // dim should be irrelevant
// ->Args({1 << G, 1 << K})
// ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, Variant::CudaMemcpy, ILP(1), TPB(1024), ATh(32) ,GS(1)) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(GPU_BM, Variant::CudaMemset, ILP(1), TPB(1024), ATh(32), GS(1)) // dim should be irrelevant
->Args({1 << G, 1 << K})
->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();