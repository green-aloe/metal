#include <metal_stdlib>

using namespace metal;

kernel void transfer3D(constant float *input, device float *result, uint3 pos [[thread_position_in_grid]], uint3 grid_size [[threads_per_grid]]) {
    int index = (pos.x * grid_size.y * grid_size.z) + (pos.y * grid_size.z) + pos.z;
    result[index] = input[index];
}