#include <metal_stdlib>

using namespace metal;

kernel void transfer1D(constant float *input, device float *result, uint pos [[thread_position_in_grid]]) {
    int index = pos;
    result[index] = input[index];
}