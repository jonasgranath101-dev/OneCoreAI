/*

    Logic & algorithms. For data training.

*/

// calc_packed.c - Download PackedArray.h/c from https://github.com/gpakosz/PackedArray
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "PackedArray.h"

int main() {
    const size_t N = 1000;
    const uint32_t bits = 10;  // Max value 1023

    // PackedArray setup
    PackedArray pa;
    PackedArray_init(&pa, N, bits);
    
    // Fill with sequential values (e.g., vertex indices)
    for (size_t i = 0; i < N; i++) {
        PackedArray_set(&pa, i, (uint32_t)(i % (1u << bits)));
    }
    
    // Calculate sum
    uint64_t sum_packed = 0;
    clock_t start = clock();
    for (size_t i = 0; i < N; i++) {
        sum_packed += PackedArray_get(&pa, i);
    }
    double time_packed = (double)(clock() - start) / CLOCKS_PER_SEC;
    
    PackedArray_destroy(&pa);
    
    // Plain array comparison
    uint32_t* plain = malloc(N * sizeof(uint32_t));
    for (size_t i = 0; i < N; i++) plain[i] = (uint32_t)(i % (1u << bits));
    
    uint64_t sum_plain = 0;
    start = clock();
    for (size_t i = 0; i < N; i++) {
        sum_plain += plain[i];
    }
    double time_plain = (double)(clock() - start) / CLOCKS_PER_SEC;
    free(plain);
    
    printf("PackedArray sum: %llu (time: %.3f ms, memory: %zu bytes)
", 
           sum_packed, time_packed * 1000, pa.buffer_size * sizeof(uint32_t));
    printf("Plain array sum: %llu (time: %.3f ms, memory: %zu bytes)
", 
           sum_plain, time_plain * 1000, N * sizeof(uint32_t));
    
    return 0;
}