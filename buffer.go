//go:build darwin

package metal

/*
#cgo CFLAGS: -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "Metal.h"
*/
import "C"

import (
	"errors"
	"math"
	"unsafe"
)

var (
	ErrInvalidBufferId = errors.New("invalid buffer id")
)

// A BufferId references a specific metal buffer created with NewBuffer*.
type BufferId int32

// Valid checks whether or not the buffer Id is valid and can be used to run a computational process
// on the GPU.
func (id BufferId) Valid() bool {
	return id > 0
}

// A BufferType is a type that can be used to create a new metal buffer.
//
// Only types up to 32 bits wide are allowed. 64-bit types (int64, uint64, float64) are deliberately
// excluded: the Metal Shading Language has no portable 64-bit scalar that a kernel could declare to
// read such a buffer back as a typed array, so allowing them would only let callers allocate memory
// no shader could meaningfully consume. Use float32/int32/uint32 (or narrower) and, if you need
// more range, split the value across multiple elements in the shader.
type BufferType interface {
	~int8 | ~int16 | ~int32 | ~uint8 | ~uint16 | ~uint32 | ~float32
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
	if err := Available(); err != nil {
		return 0, nil, err
	}

	if width < 1 {
		return 0, nil, errors.New("invalid width")
	}

	// Cap the byte count at MaxInt32 (~2 GB). buffer_new takes a size_t, so this is not a type
	// limit of the C boundary; it is a deliberate ceiling on a single buffer. Checking width
	// against the limit before multiplying also guarantees the width*size product cannot silently
	// overflow Go's int on the way to the bounds check.
	if width > math.MaxInt32/sizeof[T]() {
		return 0, nil, errors.New("exceeded maximum number of bytes")
	}
	numBytes := width * sizeof[T]()

	// The C side may strdup an error message into err on failure; we must free it.
	var err *C.char
	defer func() { freeCString(err) }()

	// Allocate memory for the new buffer and get its contents pointer in one call.
	var contents unsafe.Pointer
	bufferId := C.buffer_new(C.size_t(numBytes), &contents, &err)
	if int(bufferId) == 0 {
		return 0, nil, metalErrToError(err, "unable to create buffer", ErrInvalidBufferId)
	}

	// Wrap the buffer in a go slice.
	slice := unsafe.Slice((*T)(contents), width)

	return BufferId(bufferId), slice, nil
}

// NewBufferWith is the same as NewBuffer, but it also initializes the buffer with the provided data.
// Like NewBuffer, it requires a width of at least one, so empty data returns an "invalid width" error.
func NewBufferWith[T BufferType](data []T) (BufferId, []T, error) {
	bufferId, buffer, err := NewBuffer[T](len(data))
	if err != nil {
		return 0, nil, err
	}

	copy(buffer, data)

	return bufferId, buffer, nil
}

// Close releases the buffer from the GPU memory. The buffer Id becomes invalid after this call.
//
// Close has a pointer receiver because it zeroes the id in place to mark it invalid, so it must be
// called on an addressable BufferId (a variable, not a literal or map element). For example,
// bufferId.Close() works when bufferId is a variable, but BufferId(7).Close() does not compile.
// This differs from Function.Close, which already operates on a *Function and so reads naturally on
// a function handle.
func (id *BufferId) Close() error {
	if id == nil || !id.Valid() {
		return ErrInvalidBufferId
	}

	// The C side may strdup an error message into err on failure; we must free it.
	var err *C.char
	defer func() { freeCString(err) }()

	if !C.buffer_close(C.int(*id), &err) {
		return metalErrToError(err, "unable to free buffer", ErrInvalidBufferId)
	}

	// Clear the buffer Id to mark that it's no longer valid.
	*id = 0

	return nil
}
