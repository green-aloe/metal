#include <metal_stdlib>

using namespace metal;

kernel void transferType(constant %s *input, device %s *result, uint pos [[thread_position_in_grid]]) {
    int index = pos;
    result[index] = input[index];
}