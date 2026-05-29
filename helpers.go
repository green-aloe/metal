//go:build darwin

package metal

/*
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
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

// The error categories the C layer reports through its errorCode out-param. These must stay in
// sync with enum MetalErrorCode in Error.h. errCodeNone (the zero value) means the failure has no
// associated sentinel; it is what every out-param starts at and what plain errors leave behind.
const (
	errCodeNone              = 0
	errCodeInvalidFunctionId = 1
	errCodeInvalidBufferId   = 2
)

// sentinelForCode maps a C error code to the Go sentinel callers test for with errors.Is. An
// unrecognized or none code maps to nil (no sentinel).
func sentinelForCode(code C.int) error {
	switch int(code) {
	case errCodeInvalidFunctionId:
		return ErrInvalidFunctionId
	case errCodeInvalidBufferId:
		return ErrInvalidBufferId
	default:
		return nil
	}
}

// metalErrToError builds a Go error from a C error message and category code. metalErr is the
// human-readable message (C.GoString returns "" for a nil pointer); wrap is an optional Go-side
// prefix; code is the category the C layer set, used to attach a sentinel so callers can match
// with errors.Is. The message and the sentinel are independent: the sentinel is decided by the
// code alone, never by parsing the message text.
//
// It returns nil only when there is no message and no wrap.
func metalErrToError(metalErr *C.char, wrap string, code C.int) error {
	msg := C.GoString(metalErr)

	var err error
	switch {
	case msg == "" && wrap == "":
		return nil
	case msg == "":
		err = errors.New(wrap)
	case wrap == "":
		err = errors.New(msg)
	default:
		err = fmt.Errorf("%s: %w", wrap, errors.New(msg))
	}

	if sentinel := sentinelForCode(code); sentinel != nil {
		// Attach the sentinel for errors.Is without changing the message Error() reports (the
		// sentinel's own text is usually already a substring of msg, so joining it would duplicate).
		err = sentinelError{err: err, sentinel: sentinel}
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
