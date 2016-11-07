#include <benchmark/benchmark.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <memory>

using namespace std;


void BM_gather_materialize(benchmark::State& state) {
  size_t fact_table_size = state.range(0) >> 2; // convert to # ints
  size_t dim_table_size = state.range(1) >> 2; // convert to # ints

  assert(RAND_MAX >= dim_table_size);
  
  auto fact_table_fk_column = make_unique<uint32_t[]>(fact_table_size);
  auto dim_table_column = make_unique<uint32_t[]>(dim_table_size);
  auto output_column = make_unique<uint32_t[]>(fact_table_size);

  // init: need fk to be somewhat random into the whole of dim table.
  unsigned int sd = 0xabcd;
  for (size_t i = 0; i < fact_table_size; ++i){
    fact_table_fk_column[i] = (rand_r(&sd) % dim_table_size);
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


// marsaglia xorshift used (faster than rand_r)
inline uint32_t xorshift64star(uint64_t* x) {
    *x ^= *x >> 12; // a
    *x ^= *x << 25; // b
    *x ^= *x >> 27; // c
    return (uint32_t)(*x * UINT32_C(213338717));
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
    
  // accesses a workspace of the size |dim_table_column| in an unpredictable way..
  uint64_t total = 0;
  while (state.KeepRunning()) {
    total = 0;
  #pragma omp parallel
    {
     uint64_t sd = 0xabcd || omp_get_thread_num();
     
     #pragma omp for                \
       reduction (+: total)
     for (size_t i = 0; i < fact_table_size; ++i) {
       auto index =  mask & xorshift64star(&sd);
       total += dim_table_column[index];
     }
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



#define G <<30
#define M <<20
#define K <<10
BENCHMARK(BM_gather_materialize)
->Args({1 G, 4 K}) // fits in L1 cache comfortably...
->Args({1 G, (4*32) K}) // only fits in L2 cache
->Args({1 G, (4*256) K}) // only fits in L3 cache
->Args({1 G, 400 M})->Unit(benchmark::kMillisecond); // requires trip to...

BENCHMARK(BM_gather_add)
->Args({1 G, 4 K}) // fits in L1 cache comfortably...
->Args({1 G, (4*32) K}) // only fits in L2 cache
->Args({1 G, (4*256) K}) // only fits in L3 cache
->Args({1 G, (4*64) M})->Unit(benchmark::kMillisecond); // requires trip to...
#undef G
#undef M
#undef K


BENCHMARK_MAIN();
