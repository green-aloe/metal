#include <metal_stdlib>

using namespace metal;

kernel void transfer2D(constant float *input, device float *result, uint2 pos [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]]) {
    int index = (pos.x * grid_size.y) + pos.y;
    result[index] = input[index];
}