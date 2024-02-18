#include <metal_stdlib>

using namespace metal;

kernel void noop(uint index [[thread_position_in_grid]]) {
}
