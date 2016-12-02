#include <benchmark/benchmark.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <memory>
#include "emmintrin.h"
#include <atomic>

using namespace std;

const int G  = 30;
const int M  = 20;
const int K  = 10;

#pragma omp declare simd
inline uint32_t xorshift_hash(uint32_t x) {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return x * UINT32_C(213338717);
}

enum Variant {
  Mat = 1,
  NoMat,
  OnlyMat,
  WriteOnly
};

template <Variant variant>
void cpuGather(const int* __restrict__ fact_table,
               const int* __restrict__ dim_table,
               int* __restrict__ output_table,
               const int64_t idx_len,
               const int64_t idx_domain) {
  const auto mask = idx_domain -1;

  switch (variant){
  case Mat:{
    #pragma omp parallel for
    for (int64_t i = 0; i < idx_len; ++i) {
      output_table[i] = dim_table[fact_table[i]];
    }

    break;
  }
  case NoMat:{
    uint64_t total = 0;
    total = 0;
    #pragma omp parallel
    {     
     #pragma omp for simd                \
       reduction (+: total)
     for (int64_t i = 0; i < idx_len; ++i) {
       auto index =  mask & xorshift_hash(i);
       total += dim_table[index];
     }
    }

    ((uint64_t*)output_table)[0] = total;
    break;
  }
 case OnlyMat: {
   #pragma omp parallel for simd
    for (int64_t i = 0; i < idx_len; ++i){
      output_table[i] = 5*fact_table[i] + 1;
    }

    break;
  }
 case WriteOnly:{
    #pragma omp parallel for simd
    for (int64_t i = 0; i < idx_len; ++i){
      auto index = mask & xorshift_hash(i);
      output_table[i] = 5*index + 1;
    }

    break;
 }
  default:{
   assert(false && "unknown variant");
  }
  }
}

using KernelT = void(const int* __restrict__ fact_table,
               const int* __restrict__ dim_table,
               int* __restrict__ output_table,
               int64_t idx_len,
               int64_t idx_domain);

template <KernelT kernel, Variant variant>
void BM(benchmark::State& state){
  int64_t fact_table_size = state.range(0) >> 2; // convert to # ints
  //fprintf(stderr, "before: %d, %d\n", state.range(0), state.range(1));
  int64_t dim_table_size = state.range(1) >> 2; // convert to # ints
  //fprintf(stderr, "fact table size (int): %ld.  dim table size (int): %ld\n", fact_table_size, dim_table_size);
  auto sm = __builtin_popcountl (dim_table_size);
  assert(sm == 1); // popcount of 1.
  const auto mask = dim_table_size - 1;

  assert(RAND_MAX >= dim_table_size);
  auto fact_table_fk_column = make_unique<int32_t[]>(fact_table_size);
  auto dim_table_column = make_unique<int32_t[]>(dim_table_size);
  auto output_column = make_unique<int32_t[]>(fact_table_size);

  // init: need fk to be somewhat random into the whole of dim table.
  #pragma omp parallel
    {
     #pragma omp for simd
      for (int64_t i = 0; i < fact_table_size; ++i) {
       fact_table_fk_column[i] = mask & xorshift_hash(i);
     }
    }

  // make dim to be something easy to compute from position so we can verify results
  #pragma omp parallel for
  for (int64_t i = 0; i < dim_table_size; ++i) {
    dim_table_column[i] = i*5 + 1;
  }

  while (state.KeepRunning()) {
    kernel(fact_table_fk_column.get(), dim_table_column.get(), output_column.get(), fact_table_size, dim_table_size);
  }

  state.SetItemsProcessed(int64_t(state.iterations())*int64_t(fact_table_size));
  
  if (variant == OnlyMat){
    state.SetBytesProcessed(int64_t(state.iterations()) *
                            int64_t(fact_table_size*sizeof(int32_t) * 2));
  }


  // verify results according to variant
    switch (variant) {
    case Mat:
    case OnlyMat:
    case WriteOnly: {
      uint32_t res_expected =0 , res_actual = 0;

#pragma omp parallel for                        \
  reduction (+: res_expected, res_actual)
      for (int64_t i = 0; i < fact_table_size; ++i) {
        res_expected += (fact_table_fk_column[i]*5 + 1);
        res_actual += output_column[i];
      }

      if (res_actual != res_expected){
        cerr << "ERROR: failed correctness check." << endl;
        cerr << res_actual << " vs. "  << res_expected << endl;
      }

      break;
    }
    case NoMat:{
      auto total = ((uint64_t*)output_column.get())[0];
      auto expectation = ((uint64_t)fact_table_size/dim_table_size)*(5*((uint64_t)dim_table_size * (dim_table_size + 1))/2 + dim_table_size);

      auto ratio = ((double)total / (double)expectation);
      auto tolerance = 0.01;
      if ( ratio - 1.0 < -tolerance ||
           ratio - 1.0 > tolerance ) {
        cerr << total << " vs expectation mismmatch =  " << expectation << endl;
      }

      break;
    }
    default:
      assert(false && "No verification available for variant");
    }
}

void radix_gather(const uint32_t * __restrict__ fact, uint32_t input_len,
                  const uint32_t * __restrict__ dimension, uint32_t dimension_len,
                  uint32_t * __restrict__ output,
                  uint32_t * __restrict__ scatter_mask);

void BM_gather_buffer(benchmark::State& state) {
  size_t fact_table_size = state.range(0) >> 2; // convert to # ints
  size_t dim_table_size = state.range(1) >> 2; // convert to # ints

  auto sm = __builtin_popcountl (dim_table_size);
  assert(sm == 1); // popcount of 1.
  const auto mask = dim_table_size - 1;

  assert(RAND_MAX >= dim_table_size);
  
  auto fact_table_fk_column = make_unique<uint32_t[]>(fact_table_size);
  auto dim_table_column = make_unique<uint32_t[]>(dim_table_size);
  auto output_column = make_unique<uint32_t[]>(fact_table_size);
  auto actual_output_column = make_unique<uint32_t[]>(fact_table_size);
  auto output_column_pos = make_unique<uint32_t[]>(fact_table_size);

  // init: need fk to be somewhat random into the whole of dim table.
  #pragma omp parallel
    {
     #pragma omp for simd
     for (size_t i = 0; i < fact_table_size; ++i) {
       fact_table_fk_column[i] = mask & xorshift_hash(i);
     }
    }

  // need dim to be something easy to compute from position.
  #pragma omp parallel for
  for (size_t i = 0; i < dim_table_size; ++i) {
    dim_table_column[i] = i*5 + 1;
  }

  // gather access pattern:
  // reads fact_table_fk column sequentially.
  // writes output_column sequentially. 
  // accesses a workspace of the size |dim_table_column| in a data dependent manne.
  // time this here...
  while (state.KeepRunning()) {
    radix_gather(fact_table_fk_column.get(), fact_table_size,
                 dim_table_column.get(), dim_table_size,
                 output_column.get(),
                 output_column_pos.get());
    
  }

  // do a scatter to reorder the output entries ( to test directly)...
  // #pragma omp parallel for
  // for (size_t i = 0; i < fact_table_size; ++i) {
  //   actual_output_column[output_column_pos[i]] = output_column[i];
  // }
  uint64_t res_expected = 0, res_actual = 0;
#pragma omp parallel for                        \
  reduction (+: res_expected, res_actual)
  for (size_t i = 0; i < fact_table_size; ++i) {
    res_expected += (fact_table_fk_column[i]*5 + 1);
    res_actual += output_column[i];
  }

  if (res_actual != res_expected){
    cerr << "ERROR: failed correctness check." << endl;
    cerr << res_actual << " vs. "  << res_expected << endl;
  }

  auto expectation = ((uint64_t)fact_table_size/dim_table_size)*(5*((uint64_t)(dim_table_size -1) * dim_table_size)/2 + dim_table_size);

  auto ratio = ((double)res_actual / (double)expectation);
  auto tolerance = 0.01;
  if ( ratio - 1.0 < -tolerance ||
       ratio - 1.0 > tolerance ) {
    cerr << "ERROR: actual " << res_actual << " vs expectation mismmatch =  " << expectation << endl;
  }

}

void radix_gather(const uint32_t * __restrict__ fact, uint32_t input_len,
                  const uint32_t * __restrict__ dimension, uint32_t dimension_len,
                  uint32_t * __restrict__ output,
                  uint32_t * __restrict__ scatter_mask)
{
  const static size_t L2_CACHE_SIZE = 256 << K; // 256KB for L2, with some extra room for the intermediate buffers.
  const static size_t LLC_CACHE_SIZE = 16 << M; // 20MB for L3, trying not to do lookups within L3.
  const static size_t THREADS = 8;
  const static size_t LLC_CACHE_SIZE_PER_THREAD = LLC_CACHE_SIZE/THREADS;
  
  const static size_t kMaxBufferEntries = 512;
  const static size_t kNumBuffers = LLC_CACHE_SIZE_PER_THREAD/(kMaxBufferEntries * sizeof(uint32_t));

  auto max_fanout_per_buffer = dimension_len*sizeof(uint32_t)/kNumBuffers;
  assert(max_fanout_per_buffer < L2_CACHE_SIZE);
  assert( __builtin_popcountl (kNumBuffers) == 1); // power of two
  
  assert(dimension_len > 0);
  auto max_dimension_bit = 31 - __builtin_clz(dimension_len);
  auto mask = kNumBuffers - 1;
  auto mask_size = __builtin_popcount(mask);
  auto starting_offset = max_dimension_bit > mask_size ? (max_dimension_bit - mask_size) : 0;
  const static uint32_t kRadixMask = mask << starting_offset;
  
  assert(1 << __builtin_popcount(kRadixMask) == kNumBuffers); 

  // shared. 
  atomic<uint32_t> global_output_idx(0);

  #pragma omp parallel
  {
  uint32_t positions[kNumBuffers]{};
  uint32_t position_buffer[kNumBuffers][kMaxBufferEntries];
  // uint32_t orig_buffer[kNumBuffers][kMaxBufferEntries];
  
  // sanity checks: don't underutilize or exceed the L2 cache for our internal buffering.
  auto constexpr intermediate_sizes = sizeof(position_buffer) + sizeof(positions);
  static_assert(true || intermediate_sizes < LLC_CACHE_SIZE_PER_THREAD, "cache sizes");
  static_assert(true || LLC_CACHE_SIZE_PER_THREAD/2 < intermediate_sizes, "cache sizes");

  auto flush_buffer = [&positions, &global_output_idx, &position_buffer, &dimension, &scatter_mask, &output](uint32_t buffer, uint32_t num_entries){
      auto out_idx = global_output_idx.fetch_add(num_entries);  

      for (size_t buf_idx = 0; buf_idx < num_entries; ++buf_idx) { 
        auto a = position_buffer[buffer][buf_idx];                 
        // auto b = orig_buffer[buffer][buf_idx];          
        output[out_idx] = dimension[a];  
        // scatter_mask[out_idx] = b;                     
        out_idx++;                
      }
      
      positions[buffer] = 0;
    };

  #pragma omp for                                          
  for (size_t i = 0; i < input_len; ++i) {
    auto buffer = (kRadixMask & fact[i]) >> starting_offset;
    assert(buffer < kNumBuffers);
    auto pos = positions[buffer]++;
    
    position_buffer[buffer][pos] = fact[i];
    // orig_buffer[buffer][pos] = i;

    // if buffer is full now, do the lookups, and flush buffer
    if (pos + 1 == kMaxBufferEntries) {
      flush_buffer(buffer, kMaxBufferEntries);
    }
  }

  // flush remaining buffer contents
  for (size_t b = 0; b < kNumBuffers; ++b) {
    if (positions[b] > 0){
      flush_buffer(b, positions[b]);
    }
  }
  
  }
}

void BM_gather_buffer_test(benchmark::State& state) {
  BM_gather_buffer(state);
}


// make sure kernel and validation variants match each
// other
#define BENCHVAR($bm, $kertemplate, $variant) \
  BENCHMARK_TEMPLATE($bm, $kertemplate<$variant>, $variant)

BENCHVAR(BM, cpuGather, Mat)->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 1 << G}})
->Unit(benchmark::kMillisecond); 

BENCHVAR(BM, cpuGather, NoMat)->RangeMultiplier(2)
->Ranges({{1<<G, 1<<G}, {1 << K, 1 << G}})
->Unit(benchmark::kMillisecond); 

BENCHVAR(BM, cpuGather, OnlyMat)->RangeMultiplier(2)
->Args({1<<G, 1<<K})->Args({1<<G, 1<<G})
->Unit(benchmark::kMillisecond); 

// BENCHVAR(BM, cpuGather, WriteOnly)->RangeMultiplier(2)
// ->Ranges({{1<<G, 1<<G}, {1 << K, 2 << K}})
// ->Unit(benchmark::kMillisecond); 

BENCHMARK(BM_gather_buffer_test)->Args({64 << K, 64 << K});

BENCHMARK_MAIN();
