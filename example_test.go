package metal_test

import (
	"fmt"
	"log"

	"github.com/green-aloe/metal"
)

func ExampleNewBuffer_oneDimension() {
	// Create a 1-dimensional buffer with a width of 100 items. This will allocate 400 bytes (100
	// items * sizeof(float32)).
	width := 100
	bufferId, buffer, err := metal.NewBuffer[float32](width)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer is used to load/retrieve data into/out of the pipeline.
	_ = buffer
}

func ExampleNewBuffer_twoDimensions() {
	// Create a 1-dimensional buffer with enough items to eventually have a 2-dimensional buffer
	// with a width of 100 items and a height of 20 items. This will allocate 8,000 bytes (100 * 20
	// * sizeof(float32)).
	width, height := 100, 20
	bufferId, buffer1D, err := metal.NewBuffer[float32](width * height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// Fold the buffer into a 2D grid of 100 items wide and 20 items tall.
	buffer2D := metal.Fold(buffer1D, width)

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer2D is used to load/retrieve data into/out of the pipeline.
	_ = buffer2D
}

func ExampleNewBuffer_threeDimensions() {
	// Create a 1-dimensional buffer with enough items to eventually have a 3-dimensional buffer
	// with a width of 100 items, a height of 20 items, and a depth of 2 items. This will allocate
	// 16,000 bytes (100 * 20 * 2 * sizeof(float32)).
	width, height, depth := 100, 20, 2
	bufferId, buffer1D, err := metal.NewBuffer[float32](width * height * depth)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// Fold the buffer into a pair of 2D grids of 100 items wide and 20 items tall.
	buffer2D := metal.Fold(buffer1D, width*height)

	// Fold the 2D grids into a 3D grid of 100 items wide, 20 items tall, and 2 items deep.
	buffer3D := metal.Fold(buffer2D, width)

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer3D is used to load/retrieve data into/out of the pipeline.
	_ = buffer3D
}

func ExampleNewFunction() {
	source := `
		#include <metal_stdlib>
		#include <metal_math>

		using namespace metal;

		kernel void sine(constant float *input, device float *result, uint pos [[thread_position_in_grid]]) {
			int index = pos;
			result[pos] = sin(input[pos]);
		}
	`

	function, err := metal.NewFunction(source, "sine")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	// function is used to run the function later.
	_ = function
}

func ExampleFunction_Run_oneDimension() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void transfer1D(constant float *input, device float *output, uint pos [[thread_position_in_grid]]) {
			int index = pos;
			output[index] = input[index];
		}
	`
		width = 3
	)

	function, err := metal.NewFunction(source, "transfer1D")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, input, err := metal.NewBuffer[float32](width)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	outputId, output, err := metal.NewBuffer[float32](width)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	for i := range input {
		input[i] = float32(i + 1)
	}

	if err := function.Run(metal.RunParameters{
		Grid: metal.Grid{
			X: width,
		},
		BufferIds: []metal.BufferId{inputId, outputId},
	}); err != nil {
		log.Fatalf("Unable to run metal function: %v", err)
	}

	fmt.Println(input)
	fmt.Println(output)
	// Output:
	// [1 2 3]
	// [1 2 3]
}

func ExampleFunction_Run_twoDimensions() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void transfer2D(constant float *input, device float *output, uint2 pos [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]]) {
			int index = (pos.x * grid_size.y) + pos.y;
			output[index] = input[index];
		}
	`
		width  = 3
		height = 3
	)

	function, err := metal.NewFunction(source, "transfer2D")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, i, err := metal.NewBuffer[float32](width * height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	input := metal.Fold(i, width)

	outputId, o, err := metal.NewBuffer[float32](width * height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	output := metal.Fold(o, width)

	for i := range input {
		for j := range input[i] {
			input[i][j] = float32(i*height + j + 1)
		}
	}

	if err := function.Run(metal.RunParameters{
		Grid: metal.Grid{
			X: width,
			Y: height,
		},
		BufferIds: []metal.BufferId{inputId, outputId},
	}); err != nil {
		log.Fatalf("Unable to run metal function: %v", err)
	}

	fmt.Println(input)
	fmt.Println(output)
	// Output:
	// [[1 2 3] [4 5 6] [7 8 9]]
	// [[1 2 3] [4 5 6] [7 8 9]]
}

func ExampleFunction_Run_threeDimensions() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void transfer3D(constant float *input, device float *result, uint3 pos [[thread_position_in_grid]], uint3 grid_size [[threads_per_grid]]) {
			int index = (pos.x * grid_size.y * grid_size.z) + (pos.y * grid_size.z) + pos.z;
			result[index] = input[index];
		}
	`
		width  = 3
		height = 3
		depth  = 3
	)

	function, err := metal.NewFunction(source, "transfer3D")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, i, err := metal.NewBuffer[float32](width * height * depth)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	input := metal.Fold(metal.Fold(i, width*height), width)

	outputId, o, err := metal.NewBuffer[float32](width * height * depth)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	output := metal.Fold(metal.Fold(o, width*height), width)

	for i := range input {
		for j := range input[i] {
			for k := range input[i][j] {
				input[i][j][k] = float32(i*height*depth + j*depth + k + 1)
			}
		}
	}

	if err := function.Run(metal.RunParameters{
		Grid: metal.Grid{
			X: width,
			Y: height,
			Z: depth,
		},
		BufferIds: []metal.BufferId{inputId, outputId},
	}); err != nil {
		log.Fatalf("Unable to run metal function: %v", err)
	}

	fmt.Println(input)
	fmt.Println(output)
	// Output:
	// [[[1 2 3] [4 5 6] [7 8 9]] [[10 11 12] [13 14 15] [16 17 18]] [[19 20 21] [22 23 24] [25 26 27]]]
	// [[[1 2 3] [4 5 6] [7 8 9]] [[10 11 12] [13 14 15] [16 17 18]] [[19 20 21] [22 23 24] [25 26 27]]]
}

func ExampleFunction_RunBatch() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void scale(constant float *factor, constant float *input, device float *output, uint pos [[thread_position_in_grid]]) {
			output[pos] = input[pos] * *factor;
		}
	`
		width = 3
	)

	function, err := metal.NewFunction(source, "scale")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	// Build three independent dispatches, each with its own input/output buffers and its own scale
	// factor. RunBatch encodes all three into a single command buffer and runs them in one trip to
	// the GPU, which is faster than calling Run three times.
	const numDispatches = 3
	params := make([]metal.RunParameters, numDispatches)
	outputs := make([][]float32, numDispatches)
	for d := range params {
		inputId, input, err := metal.NewBuffer[float32](width)
		if err != nil {
			log.Fatalf("Unable to create metal buffer: %v", err)
		}
		outputId, output, err := metal.NewBuffer[float32](width)
		if err != nil {
			log.Fatalf("Unable to create metal buffer: %v", err)
		}

		for i := range input {
			input[i] = float32(i + 1)
		}

		params[d] = metal.RunParameters{
			Grid:      metal.Grid{X: width},
			Inputs:    []float32{float32(d + 1)}, // scale by 1, then 2, then 3
			BufferIds: []metal.BufferId{inputId, outputId},
		}
		outputs[d] = output
	}

	if err := function.RunBatch(params); err != nil {
		log.Fatalf("Unable to run metal function batch: %v", err)
	}

	for _, output := range outputs {
		fmt.Println(output)
	}
	// Output:
	// [1 2 3]
	// [2 4 6]
	// [3 6 9]
}

func ExampleFunction_RunAsync() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void transfer1D(constant float *input, device float *output, uint pos [[thread_position_in_grid]]) {
			output[pos] = input[pos];
		}
	`
		width = 3
	)

	function, err := metal.NewFunction(source, "transfer1D")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, input, err := metal.NewBuffer[float32](width)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	outputId, output, err := metal.NewBuffer[float32](width)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	for i := range input {
		input[i] = float32(i + 1)
	}

	// RunAsync returns immediately, before the GPU has finished, leaving the CPU free to do other
	// work in the meantime.
	handle, err := function.RunAsync(metal.RunParameters{
		Grid:      metal.Grid{X: width},
		BufferIds: []metal.BufferId{inputId, outputId},
	})
	if err != nil {
		log.Fatalf("Unable to run metal function asynchronously: %v", err)
	}

	// ... do other CPU work here while the GPU runs ...

	// Wait blocks until the dispatch finishes. The output buffer is only valid after this returns.
	if err := handle.Wait(); err != nil {
		log.Fatalf("Unable to wait for metal function: %v", err)
	}

	fmt.Println(output)
	// Output:
	// [1 2 3]
}

func ExampleFunction_RunBatchAsync() {
	const (
		source = `
		#include <metal_stdlib>

		using namespace metal;

		kernel void transfer1D(constant float *input, device float *output, uint pos [[thread_position_in_grid]]) {
			output[pos] = input[pos];
		}
	`
		width = 3
	)

	function, err := metal.NewFunction(source, "transfer1D")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	const numDispatches = 2
	params := make([]metal.RunParameters, numDispatches)
	outputs := make([][]float32, numDispatches)
	for d := range params {
		inputId, input, err := metal.NewBuffer[float32](width)
		if err != nil {
			log.Fatalf("Unable to create metal buffer: %v", err)
		}
		outputId, output, err := metal.NewBuffer[float32](width)
		if err != nil {
			log.Fatalf("Unable to create metal buffer: %v", err)
		}

		for i := range input {
			input[i] = float32(i + 1 + d*10)
		}

		params[d] = metal.RunParameters{
			Grid:      metal.Grid{X: width},
			BufferIds: []metal.BufferId{inputId, outputId},
		}
		outputs[d] = output
	}

	// RunBatchAsync commits the whole batch as one command buffer and returns immediately. Because
	// the batch is a single command buffer, one Wait completes every dispatch in it.
	handle, err := function.RunBatchAsync(params)
	if err != nil {
		log.Fatalf("Unable to run metal function batch asynchronously: %v", err)
	}

	// ... do other CPU work here while the GPU runs the whole batch ...

	if err := handle.Wait(); err != nil {
		log.Fatalf("Unable to wait for metal function: %v", err)
	}

	for _, output := range outputs {
		fmt.Println(output)
	}
	// Output:
	// [1 2 3]
	// [11 12 13]
}

func Example() {
	width := 3
	height := 3

	source := `
		#include <metal_stdlib>

		using namespace metal;

		kernel void power(constant float *multiplier, constant int *input, device int *result, uint2 pos [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]]) {
			int index = (pos.x * grid_size.y) + pos.y;
			result[index] = input[index] * input[index] * *multiplier;
		}
	`

	function, err := metal.NewFunction(source, "power")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, i, err := metal.NewBuffer[int32](width * height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	input := metal.Fold(i, width)

	outputId, o, err := metal.NewBuffer[int32](width * height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}
	output := metal.Fold(o, width)

	for i := range input {
		for j := range input[i] {
			input[i][j] = int32(i*height + j)
		}
	}

	if err := function.Run(metal.RunParameters{
		Grid: metal.Grid{
			X: width,
			Y: height,
		},
		Inputs:    []float32{2},
		BufferIds: []metal.BufferId{inputId, outputId},
	}); err != nil {
		log.Fatalf("Unable to run metal function: %v", err)
	}

	fmt.Println(input)
	fmt.Println(output)
	// Output:
	// [[0 1 2] [3 4 5] [6 7 8]]
	// [[0 2 8] [18 32 50] [72 98 128]]
}
