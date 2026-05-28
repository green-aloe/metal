//go:build darwin

package metal

/*
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "Metal.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

var ErrInvalidFunctionId = errors.New("invalid function id")

// ErrMetalUnavailable is returned by package functions when no Metal device
// could be initialized (for example, on a machine without a supported GPU or in
// a sandbox that blocks GPU access). Use errors.Is to test for it.
var ErrMetalUnavailable = errors.New("metal is not available on this system")

// metalAvailable reports whether metal_init succeeded at package load time. It
// is written once in init (before any other goroutine can run) and only read
// afterward, so it needs no synchronization.
var metalAvailable bool

func init() {
	// Initialize the device that will be used to run the computations. A failure
	// here is not fatal: the package degrades to returning ErrMetalUnavailable
	// from its public functions so importing it on an unsupported machine does
	// not abort the process.
	metalAvailable = bool(C.metal_init())
}

// Available reports whether Metal was successfully initialized and the package
// can run computations on the GPU. If it returns an error, every other call
// into the package will fail with that same error; callers that want to degrade
// gracefully (for example, to a CPU implementation) should check this first.
func Available() error {
	if !metalAvailable {
		return ErrMetalUnavailable
	}
	return nil
}

// A Function references a specific metal function.
// It is used to run computational processes on the GPU.
type Function struct {
	id int32
}

// NewFunction sets up a new function that will run on the default GPU. It is built with the
// specified function in the provided metal code. This needs to be called only once for every
// function that will be run. It returns ErrMetalUnavailable if Metal could not be initialized.
func NewFunction(metalSource, funcName string) (*Function, error) {
	if err := Available(); err != nil {
		return nil, err
	}

	src := C.CString(metalSource)
	defer C.free(unsafe.Pointer(src))

	name := C.CString(funcName)
	defer C.free(unsafe.Pointer(name))

	// The C side may strdup an error message into err on failure; we must free it.
	var err *C.char
	defer func() { freeCString(err) }()

	id := int32(C.function_new(src, name, &err))
	if id == 0 {
		return nil, metalErrToError(err, "unable to set up metal function", ErrInvalidFunctionId)
	}

	return &Function{
		id: id,
	}, nil
}

// Valid checks whether or not the function is valid and can be used to run a computational process
// on the GPU.
func (f *Function) Valid() bool {
	return f != nil && f.id > 0
}

// String returns the name of the metal function. It returns the empty string if the function is
// not valid (nil, uninitialized, or already closed).
//
// String is NOT safe to call concurrently with Close on the same Function: Close invalidates the
// underlying handle, and a String call racing with it may observe either the name or the empty
// string. Callers that share a Function across goroutines must coordinate Close with any String or
// Run calls themselves.
func (f *Function) String() string {
	if !f.Valid() {
		return ""
	}

	// function_name strdup's the result; we must free it.
	name := C.function_name(C.int(f.id))
	defer freeCString(name)

	return C.GoString(name)
}

// A Grid specifies how many threads are needed to perform all the calculations. There should be one
// thread per calculation.
//
// Typically, this is organized as a 3-dimensional grid of threads, even if all three dimensions are
// not needed. If a dimension is not used, then it should have a size of 1; a size of 0 is also
// accepted and treated as 1, so an unset dimension does the right thing. A negative size is invalid
// and causes Run to return an error. The actual size of each dimension depends on how many
// calculations need to be performed and how the data is represented in a 3-dimensional grid. Here
// some examples:
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

// Close releases the compiled pipeline for this function. The Function becomes invalid after this
// call. It is the caller's responsibility to ensure no concurrent Run or String calls are in
// progress.
func (f *Function) Close() error {
	if f == nil || !f.Valid() {
		return ErrInvalidFunctionId
	}

	// The C side may strdup an error message into err on failure; we must free it.
	var err *C.char
	defer func() { freeCString(err) }()

	if !C.function_close(C.int(f.id), &err) {
		return metalErrToError(err, "unable to close metal function", ErrInvalidFunctionId)
	}

	f.id = 0

	return nil
}

// Run executes the computational function on the GPU. This can be called multiple times for the
// same Function Id and/or same buffers and is safe for concurrent use.
//
// A grid dimension of 0 is treated as 1 (so an unset dimension behaves as a single unit). A
// negative grid dimension is invalid and returns an error.
func (f *Function) Run(params RunParameters) error {
	// Set up the dimensions of the grid. Every dimension must be at least one unit long. A zero
	// dimension is a convenience for "unused" and clamps to 1; a negative dimension is a caller bug.
	width, err := gridDimension(params.Grid.X)
	if err != nil {
		return err
	}
	height, err := gridDimension(params.Grid.Y)
	if err != nil {
		return err
	}
	depth, err := gridDimension(params.Grid.Z)
	if err != nil {
		return err
	}

	// Get pointers to the inputs and buffer IDs. float32/C.float and int32/C.int are binary
	// compatible on all Apple platforms, so we cast directly without copying.
	var inputsPtr *C.float
	if len(params.Inputs) > 0 {
		inputsPtr = (*C.float)(unsafe.Pointer(&params.Inputs[0]))
	}

	var bufferIdsPtr *C.int
	if len(params.BufferIds) > 0 {
		bufferIdsPtr = (*C.int)(unsafe.Pointer(&params.BufferIds[0]))
	}

	// The C side may strdup an error message into cErr on failure; we must free it.
	var cErr *C.char
	defer func() { freeCString(cErr) }()

	// Run the computation on the GPU.
	ok := C.function_run(C.int(f.id), width, height, depth, inputsPtr, C.int(len(params.Inputs)), bufferIdsPtr, C.int(len(params.BufferIds)), &cErr)

	// Keep the input and buffer-id slices alive until function_run returns. The C call reads through
	// inputsPtr/bufferIdsPtr (raw pointers into the slice backing arrays), which the Go garbage
	// collector cannot see; without these the collector would be free to reclaim the slices while
	// the GPU is still reading them.
	runtime.KeepAlive(params.Inputs)
	runtime.KeepAlive(params.BufferIds)

	if !ok {
		return metalErrToError(cErr, "unable to run metal function", ErrInvalidFunctionId, ErrInvalidBufferId)
	}

	return nil
}

// gridDimension validates and normalizes a single grid dimension for the C layer. A size of 0 (an
// unused dimension) clamps to 1; a negative size is a caller error. The C side takes the dimensions
// as unsigned, so all values handed across must be positive.
func gridDimension(size int) (C.uint, error) {
	switch {
	case size < 0:
		return 0, errors.New("invalid grid dimension")
	case size == 0:
		return 1, nil
	default:
		return C.uint(size), nil
	}
}
