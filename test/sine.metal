#include <metal_stdlib>
#include <metal_math>

using namespace metal;

kernel void sine(constant float *mult, constant float *input, device float *result, uint pos [[thread_position_in_grid]]) {
    result[pos] = sin(input[pos]) * *mult * *mult;
}