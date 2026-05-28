//go:build darwin

package metal

/*
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// Fold folds a 1-dimensional slice of N items into a 2-dimensional slice of width x (N/width)
// items. Both slices have the same backing array. width must equally divide the number of items.
// All sub-slices in the returned slice have a capacity equal to N/width.
func Fold[T any](items []T, width int) [][]T {
	if len(items) == 0 || width < 1 || len(items)%width != 0 {
		return nil
	}

	height := len(items) / width

	// Partition by column: each sub-slice is a contiguous run of height items, so colStart steps
	// forward by height to mark the start of each successive column.
	plane := make([][]T, 0, width)
	for colStart := 0; colStart < len(items); colStart += height {
		plane = append(plane, items[colStart:colStart+height:colStart+height])
	}

	return plane
}

// sizeof returns the size in bytes of the generic type T.
func sizeof[T any]() int {
	var t T
	return int(unsafe.Sizeof(t))
}

// metalErrToError wraps the metal error metalErr inside wrap. C.GoString returns "" for a nil
// pointer, so a nil or empty metalErr is treated the same as "no metal error".
//
// Any of the given sentinels whose message text appears in the metal error is joined onto the
// returned error, so callers can test for ErrInvalidBufferId / ErrInvalidFunctionId with errors.Is
// while still seeing the full descriptive message. (The C layer reports a bad buffer as "invalid
// buffer id: N" and a bad function as "invalid function id: N", which contain the sentinel text.)
func metalErrToError(metalErr *C.char, wrap string, sentinels ...error) error {
	msg := C.GoString(metalErr)

	var err error
	switch {
	case msg == "" && wrap == "":
		// We have neither a metal error nor any wrapping. Return nil.
		return nil
	case msg == "":
		// We have wrapping but we don't have a metal error. Return just the wrapping.
		err = errors.New(wrap)
	case wrap == "":
		// We have a metal error but we don't have any wrapping. Return just the metal error.
		err = errors.New(msg)
	default:
		// We have both a metal error and wrapping. Return both of them formatted together.
		err = fmt.Errorf("%s: %w", wrap, errors.New(msg))
	}

	for _, sentinel := range sentinels {
		if sentinel != nil && strings.Contains(msg, sentinel.Error()) {
			// Attach the sentinel for errors.Is without altering the message text that Error() reports.
			err = sentinelError{err: err, sentinel: sentinel}
			break
		}
	}

	return err
}

// sentinelError attaches a sentinel to an existing error so errors.Is matches the sentinel while
// Error() still returns the original (descriptive) message unchanged.
type sentinelError struct {
	err      error
	sentinel error
}

func (e sentinelError) Error() string { return e.err.Error() }

// Is reports a match for the attached sentinel and defers to the wrapped error for anything else.
func (e sentinelError) Is(target error) bool { return target == e.sentinel }

func (e sentinelError) Unwrap() error { return e.err }

// freeCString releases a C string allocated on the C heap (via strdup, malloc,
// or C.CString). Safe to call with a nil pointer.
func freeCString(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// cgoString and cgoFree are wrappers around cgo functions for the test files. They must live in a
// non-test file: Go does not support `import "C"` from _test.go files ("use of cgo in test not
// supported"), so the tests reach C only through wrappers like these.
func cgoString(s string) *C.char { return C.CString(s) }
func cgoFree(s *C.char)          { C.free(unsafe.Pointer(s)) }
