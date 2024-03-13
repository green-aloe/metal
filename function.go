//go:build darwin
// +build darwin

package metal

// frameworks not included:
// Cocoa

/*
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation
#include "metal.h"
*/
import "C"

import (
	"unsafe"
)

func init() {
	// Initialize the device that will be used to run the computations.
	C.metal_init()
}

// A Function references a specific metal function.
// It is used to run computational processes on the GPU.
type Function struct {
	id int
}

// NewFunction sets up a new function that will run on the default GPU. It is built with the
// specified function in the provided metal code. This needs to be called only once for every
// function that will be run.
func NewFunction(metalSource, funcName string) (Function, error) {
	src := C.CString(metalSource)
	defer C.free(unsafe.Pointer(src))

	name := C.CString(funcName)
	defer C.free(unsafe.Pointer(name))

	err := C.CString("")
	defer C.free(unsafe.Pointer(err))

	id := int(C.function_new(src, name, &err))
	if id == 0 {
		return Function{}, metalErrToError(err, "Unable to set up metal function")
	}

	return Function{
		id: id,
	}, nil
}

// Valid checks whether or not the function is valid and can be used to run a computational process
// on the GPU.
func (function Function) Valid() bool {
	return function.id > 0
}

// String returns the name of the metal function.
func (function Function) String() string {
	if !function.Valid() {
		return ""
	}

	name := C.function_name(C.int(function.id))

	return C.GoString(name)
}

// A Grid specifies how many threads are needed to perform all the calculations. There should be one
// thread per calculation.
//
// Typically, this is organized as a 3-dimensional grid of threads, even if all three dimensions are
// not needed. If a dimension is not used, then it should have a size of 1. The actual size of each
// dimension depends on how many calculations need to be performed and how the data is represented
// in a 3-dimensional grid. Here some examples:
//
// - If the computational problem is to square a list of numbers, then only one dimension is needed:
// the list of numbers to square. If the list has 10,000 numbers in it, then one would use a (10,000
// x 1 x 1) grid. If the computational problem is to multiply one list of numbers against another
// list, then only one dimension is still needed, because there's only one operation per item in the
// list.
//
// - If the computational problem is to perform an operation on every pixel in an image, then it can
// be conceptually broken up into two dimensions, even if the list of pixels is a long,
// 1-dimensional array. If the image is 600 x 800 pixels, then one would use a (600 x 800 x 1) grid.
//
// - If the computational problem is to calculate the vector of objects in a 3-dimensional space,
// then all three dimensions in the grid can be used to represent the entire space. If the space is
// 100 units x 200 units x 300 units, then one would use a (100 x 200 x 300) grid.
//
// For more information on grid sizes, see
// https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes.
type Grid struct {
	X int
	Y int
	Z int
}

type RunParameters struct {
	// Grid that defines the dimensions of the buffers used to run the computation.
	Grid Grid
	// List of static inputs that are used to run the computation. These are not indexed by position
	// in the grid like the buffers are but are instead used as constants for every iteration. They
	// are supplied as the first arguments to the metal function in the order given here.
	Inputs []float32
	// List of buffer Ids that are used to retrieve the correct block of memory for the buffers. The
	// buffers are indexed by position in the grid. They are supplied as arguments to the metal
	// function after the inputs in the order given here.
	BufferIds []BufferId
}

// Run executes the computational function on the GPU. This can be called multiple times for the
// same Function Id and/or same buffers and is safe for concurrent use.
func (function Function) Run(params RunParameters) error {

	// Make a list of inputs, and get a pointer to the beginning of them (if we have any).
	inputs, inputsPtr := convertList[float32, C.float](params.Inputs)

	// Make a list of buffer Ids, and get a pointer to the beginning of them (if we have any).
	bufferIds, bufferIdsPtr := convertList[BufferId, C.int](params.BufferIds)

	// Set up the dimensions of the grid. Every dimension must be at least one unit long.
	width, height, depth := C.int(params.Grid.X), C.int(params.Grid.Y), C.int(params.Grid.Z)
	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}
	if depth < 1 {
		depth = 1
	}

	// Set up some space to hold a possible error message.
	err := C.CString("")
	defer C.free(unsafe.Pointer(err))

	// Run the computation on the GPU.
	if !C.function_run(C.int(function.id), width, height, depth, inputsPtr, C.int(len(inputs)), bufferIdsPtr, C.int(len(bufferIds)), &err) {
		return metalErrToError(err, "Unable to run metal function")
	}

	return nil
}
