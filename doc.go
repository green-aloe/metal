/*
Package metal is a library for running GPU compute kernels on macOS via Apple's [Metal API].

# Overview

Apple's unified memory architecture means the CPU and GPU share the same physical RAM.
This library allocates blocks of that shared memory, compiles Metal Shading Language (MSL)
shader source at runtime, and dispatches parallel computations to the GPU — all from Go.

# Usage

The typical pattern is:

 1. Call [NewFunction] once per shader to compile the MSL source and build a compute
    pipeline. This returns a [*Function] handle. Reuse it for every invocation.

 2. Call [NewBuffer] or [NewBufferWith] to allocate shared CPU/GPU memory. Each call
    returns a [BufferId] and a Go slice backed by that memory. Write inputs into the slice
    directly — no copy is needed.

 3. Call [Function.Run] with a [RunParameters] describing the grid dimensions, any static
    scalar inputs, and the buffer IDs. The GPU reads from and writes to the same memory the
    Go slices point to. Results are available in the slice immediately after Run returns.

 4. Call [BufferId.Close] and [Function.Close] when resources are no longer needed.

# Running: synchronous, batched, and asynchronous

There are four ways to dispatch work, trading simplicity for throughput:

  - [Function.Run] encodes one dispatch, commits it, and blocks until the GPU finishes.
    Simplest; results are ready when it returns.

  - [Function.RunBatch] encodes many dispatches into a single GPU command buffer and waits
    once for the whole batch. This amortizes the per-command-buffer setup and the single
    CPU/GPU synchronization across all dispatches, so it is much faster than calling Run in a
    loop when you have many independent dispatches to run back to back.

  - [Function.RunAsync] encodes and commits one dispatch but returns a [*RunHandle]
    immediately without waiting, so the CPU is free to encode more work or do other
    computation while the GPU runs. Call [RunHandle.Wait] exactly once to block for
    completion; the output buffers are only valid after Wait returns, and the dispatch's
    buffers must not be closed before then.

  - [Function.RunBatchAsync] combines the two: it commits a whole batch as one command
    buffer and returns a [*RunHandle] without waiting. A single [RunHandle.Wait] completes
    the entire batch.

Apple recommends minimizing the number of command buffers and avoiding unnecessary CPU/GPU
synchronization; the batched and async variants exist for workloads where the per-Run round
trip dominates.

# Concurrency

[Function.Run], [Function.RunBatch], [Function.RunAsync], and [Function.RunBatchAsync] are
all safe for concurrent use — multiple goroutines may dispatch on the same [*Function]
simultaneously. [NewFunction] and [NewBuffer] are also safe for concurrent use.
[BufferId.Close] and [Function.Close] are NOT safe to call concurrently with a dispatch on
the same resource; for the async variants this means the buffers must stay open until
[RunHandle.Wait] returns.

# Buffers and dimensions

Buffers are always allocated as a flat 1D slice. Use [Fold] to create a 2D or 3D view over
the same memory without copying:

	// 2D: width columns, height rows
	_, flat, _ := metal.NewBuffer[float32](width * height)
	grid := metal.Fold(flat, width)  // grid[x][y]

	// 3D: width × height × depth
	_, flat, _ = metal.NewBuffer[float32](width * height * depth)
	grid3D := metal.Fold(metal.Fold(flat, width*height), width)  // grid3D[x][y][z]

[Fold] partitions by column: Fold(buf, width) produces width sub-slices each of length
N/width, so grid[x][y] maps to flat index x*(N/width)+y.

# Static scalar inputs

[RunParameters].Inputs passes constant scalar values to the shader without allocating a
buffer. They are passed as the first arguments to the kernel, before the buffers. Go always
sends them as float32 bits; the Metal shader's parameter type governs interpretation.

# Types

This is the mapping of Go types to Metal types:

	| Go      | Metal  |
	| ------- | ------ |
	| float32 | float  |
	| N/A     | half   |
	| int32   | int    |
	| int16   | short  |
	| uint32  | uint   |
	| uint16  | ushort |

# Limitations

  - macOS only. The package does not compile on other platforms (all files are
    constrained to //go:build darwin).
  - Requires a Metal-capable GPU. On hardware with non-uniform threadgroup support
    (Apple4 and later, or the Mac2 family) the grid is dispatched exactly; on other
    Metal GPUs it falls back to rounded-up threadgroup dispatch, and kernels must
    bounds-check their thread position. See [page 4 here] for a compatibility table.
  - All buffers use shared CPU/GPU memory (MTLResourceStorageModeShared). GPU-private
    buffers are not supported.
  - Only compute kernels are supported (kernel void functions). Vertex and fragment
    shaders are not.
  - MSL source is compiled at runtime. Pre-compiled .metallib files are not supported.

# Resources

  - https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
  - https://adrianhesketh.com/2022/03/31/use-m1-gpu-with-go/

The Metal Shading Language specification is at [MSL Specification].

[Metal API]: https://developer.apple.com/metal/
[page 4 here]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[MSL Specification]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
*/
package metal
