//go:build darwin
// +build darwin

package metal

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Fold folds a 1-dimensional slice of N items into a 2-dimensional slice of width x (N/width)
// items. width must equally divide the number of items. All sub-slices in the returned slice have a
// capacity equal to N/width.
func Fold[T any](items []T, width int) [][]T {
	if len(items) == 0 || width < 1 || len(items)%width != 0 {
		return nil
	}

	height := len(items) / width

	plane := make([][]T, 0, width)
	for start := 0; start < len(items); start += height {
		plane = append(plane, items[start:start+height:start+height])
	}

	return plane
}

// sizeof returns the size in bytes of the generic type T.
func sizeof[T any]() int {
	var t T
	return int(unsafe.Sizeof(t))
}

// convertList converts a list from one type to another by typecasting the elements. It returns the
// converted list and a pointer to the first element. If the list is empty, this returns nil, nil.
func convertList[I, O BufferType](inputs []I) ([]O, *O) {
	if len(inputs) == 0 {
		return nil, nil
	}

	// Convert the inputs from one type to another.
	var outputs []O
	for i := range inputs {
		outputs = append(outputs, O(inputs[i]))

	}

	// Get a pointer to the first element of the outputs.
	outputsPtr := &outputs[0]

	return outputs, outputsPtr
}

// metalErrToError wraps the metal error metalErr inside wrap.
func metalErrToError(metalErr *C.char, wrap string) error {
	switch {
	case metalErr == nil || C.strlen(metalErr) == 0:
		if wrap == "" {
			// We have neither a metal error nor any wrapping. Return nil.
			return nil
		}

		// We have wrapping but we don't have a metal error. Return just the wrapping.
		return errors.New(wrap)

	default:
		mErr := errors.New(C.GoString(metalErr))

		if wrap == "" {
			// We have a metal error but we don't have any wrapping. Return just the metal error.
			return mErr
		}

		// We have both a metal error and wrapping. Return both of them formatted together.
		return fmt.Errorf("%s: %w", wrap, mErr)
	}
}

// cgoString and cgoFree are wrappers around cgo functions to enable using cgo in test files.
func cgoString(s string) *C.char { return C.CString(s) }
func cgoFree(s *C.char)          { C.free(unsafe.Pointer(s)) }
