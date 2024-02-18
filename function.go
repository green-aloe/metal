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

// A FunctionId references a specific metal function created with NewFunction. It is used to run
// computational processes with that function.
type FunctionId int

// NewFunction sets up a new function that will run on the default GPU. It is built with the
// specified function in the provided metal code. This needs to be called only once for every
// function.
func NewFunction(metalSource, funcName string) (FunctionId, error) {
	src := C.CString(metalSource)
	defer C.free(unsafe.Pointer(src))

	name := C.CString(funcName)
	defer C.free(unsafe.Pointer(name))

	err := C.CString("")
	defer C.free(unsafe.Pointer(err))

	id := int(C.function_new(src, name, &err))
	if id == 0 {
		return 0, metalErrToError(err, "Unable to set up metal function")
	}

	return FunctionId(id), nil
}

// Valid checks whether or not the function Id is valid and can be used to run a computational
// process on the GPU.
func (id FunctionId) Valid() bool {
	return id > 0
}

// String returns the name of the metal function.
func (id FunctionId) String() string {
	if !id.Valid() {
		return ""
	}

	name := C.function_name(C.int(id))

	return C.GoString(name)
}

// A Grid specifies how many threads we need to perform all the calculations. There should be one
// thread per calculation.
//
// Typically, this is organized as a 3-dimensional grid of threads, even if all three dimensions are
// not needed. If a dimension is not used, then it should have a size of 1. The actual size of each
// dimension depends on how many calculations need to be performed and how the data is represented
// in a 3-dimensional grid. Here some examples:
//
// - If the computational problem is to square a list of numbers, then we need only one dimension:
// the list of numbers to square. If the list has 10,000 numbers in it, then we would use a (10,000
// x 1 x 1) grid. If the computational problem is to multiple one list of numbers against another
// list, then we still need only one dimension, because there's only one operation per item in the
// list.
//
// - If the computational problem is to perform an operation on every pixel in an image, then we can
// conceptually break that up into two dimensions, even if the list of pixels is a long,
// 1-dimensional array. If the image is 600 x 800 pixels, then we would use a (600 x 800 x 1) grid.
//
// - If the computational problem is to calculate the vector of objects in a 3-dimensional space,
// then we can use all three dimensions in the grid to represent the entire space. If the space is
// 100 units x 200 units x 300 units, then we would use a (100 x 200 x 300) grid.
//
// For more information on grid sizes, see
// https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes.
type Grid struct {
	X int
	Y int
	Z int
}

// Run executes the computational function on the GPU. buffers is a list of buffers that have a
// buffer Id, which is used to retrieve the correct block of memory for the buffer. Each buffer is
// supplied as an argument to the metal function in the order given here. This can be called
// multiple times for the same Function Id and/or same buffers and is safe for concurrent use.
func (id FunctionId) Run(grid Grid, buffers ...BufferId) error {

	// Make a list of buffer Ids.
	var bufferIds []C.int
	for _, buffer := range buffers {
		bufferIds = append(bufferIds, C.int(buffer))
	}

	// Get a pointer to the beginning of the list of buffer Ids (if we have any).
	var bufferPtr *C.int
	if len(bufferIds) > 0 {
		bufferPtr = &bufferIds[0]
	}

	// Set up the dimensions of the grid. Every dimension must be at least one unit long.
	width, height, depth := C.int(grid.X), C.int(grid.Y), C.int(grid.Z)
	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}
	if depth < 1 {
		depth = 1
	}

	err := C.CString("")
	defer C.free(unsafe.Pointer(err))

	// Run the computation on the GPU.
	if ok := C.function_run(C.int(id), width, height, depth, bufferPtr, C.int(len(bufferIds)), &err); !ok {
		return metalErrToError(err, "Unable to run metal function")
	}

	return nil
}
