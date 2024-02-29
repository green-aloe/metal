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
	"errors"
	"math"
	"unsafe"
)

// A BufferId references a specific metal buffer created with NewBuffer*.
type BufferId int32

// Valid checks whether or not the buffer Id is valid and can be used to run a computational process
// on the GPU.
func (id BufferId) Valid() bool {
	return id > 0
}

// A BufferType is a type that can be used to create a new metal buffer.
type BufferType interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

// NewBuffer1D allocates a 1-dimensional block of memory that is accessible to both the CPU and GPU.
// It returns a unique Id for the buffer and a slice that wraps the new memory and has a length and
// capacity equal to width. This should be called only once for every argument supplied to a metal
// function, no matter how many times the buffer is used or for which metal functions.
//
// The Id is used to reference the buffer as an argument for the metal function.
//
// Only the contents of the slice should be modified. Its length and capacity and the pointer to its
// underlying array should not be altered.
func NewBuffer1D[T BufferType](width int) (BufferId, []T, error) {
	return newBuffer[T](width)
}

// NewBuffer2D allocates a 2-dimensional block of memory that is accessible to both the CPU and GPU.
// It returns a unique Id for the buffer and a slice that wraps the new memory and has a length and
// capacity equal to width. Each element in the slice is another slice with a length equal to
// height. This should be called only once for every argument supplied to a metal function, no
// matter how many times the buffer is used or for which metal functions.
//
// The Id is used to reference the buffer as an argument for the metal function.
//
// Only the contents of the slices should be modified. Their lengths and capacities and the pointers
// to their underlying arrays should not be altered.
func NewBuffer2D[T BufferType](width, height int) (BufferId, [][]T, error) {
	bufferId, b1, err := newBuffer[T](width, height)
	if err != nil {
		return 0, nil, err
	}

	b2 := fold(b1, width)

	return bufferId, b2, nil
}

// NewBuffer3D allocates a 3-dimensional block of memory that is accessible to both the CPU and GPU.
// It returns a unique Id for the buffer and a slice that wraps the new memory and has a length and
// capacity equal to width. Each element in the slice is another slice with a length equal to
// height, and each of their elements is in turn another slice with a length equal to depth. This
// should be called only once for every argument supplied to a metal function, no matter how many
// times the buffer is used or for which metal functions.
//
// The Id is used to reference the buffer as an argument for the metal function.
//
// Only the contents of the slices should be modified. Their lengths and capacities and the pointers
// to their underlying arrays should not be altered.
func NewBuffer3D[T BufferType](width, height, depth int) (BufferId, [][][]T, error) {
	bufferId, b1, err := newBuffer[T](width, height, depth)
	if err != nil {
		return 0, nil, err
	}

	b2 := fold(b1, width*height)
	b3 := fold(b2, width)

	return bufferId, b3, nil
}

// newBuffer is the common internal function for creating a new buffer with N dimensions.
func newBuffer[T BufferType](dimLens ...int) (BufferId, []T, error) {
	if len(dimLens) == 0 {
		return 0, nil, errors.New("Missing dimension(s)")
	}

	// Calculate how many elements we'll need based on the dimensions provided, and also check that
	// each dimension is valid and won't overflow the maximum bytes.
	numElems := 1
	numBytes := sizeof[T]()
	for _, dimLen := range dimLens {
		if dimLen < 1 {
			return 0, nil, errors.New("Invalid dimension")
		}

		numElems *= dimLen
		numBytes *= dimLen
		if numBytes > math.MaxInt32 || numBytes < 0 {
			return 0, nil, errors.New("Exceeded maximum number of bytes")
		}
	}

	err := C.CString("")
	defer C.free(unsafe.Pointer(err))

	// Allocate memory for the new buffer.
	bufferId := C.buffer_new(C.int(numBytes), &err)
	if int(bufferId) == 0 {
		return 0, nil, metalErrToError(err, "Unable to create buffer")
	}

	// Retrieve a pointer to the beginning of the new memory using the buffer's Id.
	newBuffer := C.buffer_retrieve(bufferId, &err)
	if newBuffer == nil {
		return 0, nil, metalErrToError(err, "Unable to retrieve buffer")
	}

	// Wrap the buffer in a go slice.
	slice := unsafe.Slice((*T)(newBuffer), numElems)

	return BufferId(bufferId), slice, nil
}
