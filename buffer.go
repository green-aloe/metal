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

// NewBuffer allocates a 1-dimensional block of memory that is accessible to both the CPU and GPU.
// It returns a unique Id for the buffer and a slice that wraps the new memory and has a length and
// capacity equal to width. The buffer is safe for reuse with any metal function.
//
// The Id is used to reference the buffer as an argument for the metal function.
//
// Only the contents of the slice should be modified. Its length and capacity and the pointer to its
// underlying array should not be altered.
//
// Use Fold to safely portion the slice into more dimensions. For example, to convert a
// one-dimensional slice into a two-dimensional slice, use Fold(buffer, width). Or to go from one
// dimensions to three, use Fold(Fold(buffer, width*height), width).
func NewBuffer[T BufferType](width int) (BufferId, []T, error) {
	if width < 1 {
		return 0, nil, errors.New("Invalid width")
	}

	numBytes := width * sizeof[T]()
	if numBytes > math.MaxInt32 || numBytes < 0 {
		return 0, nil, errors.New("Exceeded maximum number of bytes")
	}

	// Set up some space to hold a possible error message.
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
	slice := unsafe.Slice((*T)(newBuffer), width)

	return BufferId(bufferId), slice, nil
}

// NewBufferWith is the same as NewBuffer, but it also initializes the buffer with the provided data.
func NewBufferWith[T BufferType](data []T) (BufferId, []T, error) {
	bufferId, buffer, err := NewBuffer[T](len(data))
	if err != nil {
		return 0, nil, err
	}

	for i := range data {
		buffer[i] = data[i]
	}

	return bufferId, buffer, nil
}
