package metal_test

import (
	"fmt"
	"log"

	"github.com/green-aloe/metal"
)

func ExampleNewBuffer1D() {
	// Create a 1-dimensional buffer with a width of 100 items. This will allocate 400 bytes (100
	// items * sizeof(float32)).
	bufferId, buffer, err := metal.NewBuffer1D[float32](100)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer is used to load/retrieve data from the pipeline.
	_ = buffer
}

func ExampleNewBuffer2D() {
	// Create a 2-dimensional buffer with a width of 100 items and a height of 20 items. This will
	// allocate 8,000 bytes (100 * 20 * sizeof(float32)).
	bufferId, buffer, err := metal.NewBuffer2D[float32](100, 20)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer is used to load/retrieve data from the pipeline.
	_ = buffer
}

func ExampleNewBuffer3D() {
	// Create a 3-dimensional buffer with a width of 100 items, a height of 20 items, and a depth of
	// 2 items. This will allocate 16,000 bytes (100 * 20 * 2 * sizeof(float32)).
	bufferId, buffer, err := metal.NewBuffer3D[float32](100, 20, 2)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	// bufferId is used to reference the buffer when running a metal function later.
	_ = bufferId

	// buffer is used to load/retrieve data from the pipeline.
	_ = buffer
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

	setInput := func(i, j int) int32 {
		return int32(i*height + j)
	}

	source := `
		#include <metal_stdlib>

		using namespace metal;

		kernel void power(constant int *input, device int *result, uint2 pos [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]]) {
			int index = (pos.x * grid_size.y) + pos.y;
			result[index] = input[index] * input[index];
		}
	`

	function, err := metal.NewFunction(source, "power")
	if err != nil {
		log.Fatalf("Unable to create metal function: %v", err)
	}

	inputId, input, err := metal.NewBuffer2D[int32](width, height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	outputId, output, err := metal.NewBuffer2D[int32](width, height)
	if err != nil {
		log.Fatalf("Unable to create metal buffer: %v", err)
	}

	for i := range input {
		for j := range input[i] {
			input[i][j] = setInput(i, j)
		}
	}

	grid := metal.Grid{
		X: width,
		Y: height,
	}
	if err := function.Run(grid, nil, []metal.BufferId{inputId, outputId}); err != nil {
		log.Fatalf("Unable to run metal function: %v", err)
	}

	fmt.Println(input)
	fmt.Println(output)
	// Output:
	// [[0 1 2] [3 4 5] [6 7 8]]
	// [[0 1 4] [9 16 25] [36 49 64]]
}
