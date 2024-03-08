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

	// buffer is used to load/retrieve data from the pipeline.
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

	// buffer2D is used to load/retrieve data from the pipeline.
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

	// buffer3D is used to load/retrieve data from the pipeline.
	_ = buffer3D
}

func ExampleNewFunction() {
	source := `
		#include <metal_stdlib>
		#include <metal_math>

		using namespace metal;

		kernel void sine(constant float *input, device float *result, uint pos [[thread_position_in_grid]]) {
			int index = pos;
			result[pos] = sin(input[pos]) * 0.01 * 0.01;
		}
	`

	function, err := metal.NewFunction(source, "sine")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	// function is used to run the function later.
	_ = function
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
