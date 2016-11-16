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


const uint64_t G  = 30;
const uint64_t M  = 20;
const uint64_t K  = 10;


#pragma omp declare simd
inline uint32_t xorshift_hash(uint32_t x) {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return x * UINT32_C(213338717);
}

void BM_gather_materialize(benchmark::State& state) {
  size_t fact_table_size = state.range(0) >> 2; // convert to # ints
  size_t dim_table_size = state.range(1) >> 2; // convert to # ints

  auto sm = __builtin_popcountl (dim_table_size);
  assert(sm == 1); // popcount of 1.
  const auto mask = dim_table_size - 1;

  assert(RAND_MAX >= dim_table_size);
  
  auto fact_table_fk_column = make_unique<uint32_t[]>(fact_table_size);
  auto dim_table_column = make_unique<uint32_t[]>(dim_table_size);
  auto output_column = make_unique<uint32_t[]>(fact_table_size);

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
    #pragma omp parallel for
    for (size_t i = 0; i < fact_table_size; ++i) {
      output_column[i] = dim_table_column[fact_table_fk_column[i]];
    }
    
  }

  uint32_t res_expected =0 , res_actual = 0;
  #pragma omp parallel for \
    reduction (+: res_expected, res_actual)
  for (size_t i = 0; i < fact_table_size; ++i) {
    res_expected += (fact_table_fk_column[i]*5 + 1);
    res_actual += output_column[i];
  }

  if (res_actual != res_expected){
    cerr << "ERROR: failed correctness check." << endl;
    cerr << res_actual << " vs. "  << res_expected << endl;
  }
}



void BM_gather_add(benchmark::State& state) {
  size_t fact_table_size = state.range(0) >> 2;
  size_t dim_table_size = state.range(1) >> 2; // convert to # ints
      
  auto sm = __builtin_popcountl (dim_table_size);
  assert(sm == 1); // popcount of 1.
  const auto mask = dim_table_size - 1;

  assert(RAND_MAX >= dim_table_size);
  
  auto dim_table_column = make_unique<uint32_t[]>(dim_table_size);

  // need dim to be something easy to compute from position.
  #pragma omp parallel for
  for (size_t i = 0; i < dim_table_size; ++i) {
    dim_table_column[i] = i*5 + 1;
  }

  {
    int busy_loop = 0;
    int MAX_ITER = 1<<28; // 0.5 G * ~5 cycles * / 3GHz ~ 0.5 sec pause.
    while (busy_loop < MAX_ITER){
      _mm_pause ();
      _mm_pause ();
      ++busy_loop;
    }
  }
  
  // accesses a workspace of the size |dim_table_column| in an unpredictable way..
  uint64_t total = 0;
  while (state.KeepRunning()) {
    total = 0;
  #pragma omp parallel
    {     
     #pragma omp for simd                \
       reduction (+: total)
     for (size_t i = 0; i < fact_table_size; ++i) {
       auto index =  mask & xorshift_hash(i);
       total += dim_table_column[index];
     }
    }
  }


  {
    int busy_loop = 0;
    int MAX_ITER = 1<<29; // 0.5 G * ~5 cycles * / 3GHz ~ 0.5 sec pause.
    while (busy_loop < MAX_ITER){
      _mm_pause ();
      ++busy_loop;
    }
  }

  auto expectation = ((uint64_t)fact_table_size/dim_table_size)*(5*((uint64_t)dim_table_size * (dim_table_size + 1))/2 + dim_table_size);


  auto ratio = ((double)total / (double)expectation);
  auto tolerance = 0.01;
  if ( ratio - 1.0 < -tolerance ||
       ratio - 1.0 > tolerance ) {
      cerr << total << " vs expectation mismmatch =  " << expectation << endl;
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
  const static size_t L2_CACHE_SIZE =  256 << K; // 256KB for L2, with some extra room for the intermediate buffers.
  const static size_t LLC_CACHE_SIZE = 16 << M; // 20MB for L3, trying not to do lookups within L3.
  const static size_t THREADS = 8;
  const static size_t LLC_CACHE_SIZE_PER_THREAD = LLC_CACHE_SIZE/THREADS;
  
  const static size_t kMaxBufferEntries = 256;
  const static size_t kNumBuffers = LLC_CACHE_SIZE_PER_THREAD/(kMaxBufferEntries * sizeof(uint32_t));

  auto max_fanout_per_buffer = dimension_len*sizeof(uint32_t)/kNumBuffers;
  assert(max_fanout_per_buffer < L2_CACHE_SIZE);
  assert( __builtin_popcountl (kNumBuffers) == 1); // power of two
  
  assert(dimension_len > 0);
  auto max_dimension_bit = 31 - __builtin_clz(dimension_len);
  auto mask = kNumBuffers - 1;
  auto mask_size = __builtin_popcount(mask);
  auto starting_offset = max_dimension_bit > mask_size? (max_dimension_bit - mask_size) : 0;
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
  static_assert(intermediate_sizes < LLC_CACHE_SIZE_PER_THREAD, "cache sizes");
  static_assert(LLC_CACHE_SIZE_PER_THREAD/2 < intermediate_sizes, "cache sizes");

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

BENCHMARK(BM_gather_materialize)
->Args({1 << G, 4 << K}) // fits in L1 cache comfortably...
->Args({1 << G, (4*32) << K}) // only fits in L2 cache
->Args({1 << G, (4*256) << K}) // only fits in L3 cache
->Args({1 << G, 16 << M})  // fits in LLC cache with less room...
->Args({1 << G, (4*16) << M})
->Args({1 << G, (4*64) << M}) // only fits in L3 cache
->Args({1 << G, (8*64) << M})
->Args({1 << G, 1 << G})
->Unit(benchmark::kMillisecond); // requires trip to...

BENCHMARK(BM_gather_add)
->Args({1 << G, 4 << K}) // fits in L1 cache comfortably...
->Args({1 << G, (4*32) << K}) // fully fits in L2 cache...
->Args({1 << G, (4*256) << K}) // only fits in L3 cache
->Args({1 << G, 16 << M})  // fits in LLC cache...
->Args({1 << G, (4*16) << M})
->Args({1 << G, (4*64) << M})
->Args({1 << G, (8*64) << M})
->Args({1 << G, 1 << G})
->Unit(benchmark::kMillisecond); // requires trip to...


BENCHMARK(BM_gather_buffer)
->Args({1 << G, 4 << K}) // fits in L1 cache comfortably...
->Args({1 << G, (4*32) << K}) // fully fits in L2 cache...
->Args({1 << G, (4*256) << K}) // only fits in L3 cache
->Args({1 << G, 16 << M})  // fits in LLC cache...
->Args({1 << G, (4*16) << M})
->Args({1 << G, (4*64) << M})
->Args({1 << G, (8*64) << M})
->Args({1 << G, 1 << G})
->Unit(benchmark::kMillisecond); // requires trip to...

BENCHMARK(BM_gather_buffer_test)->Args({64 << K, 64 << K});

BENCHMARK_MAIN();
