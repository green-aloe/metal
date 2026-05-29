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
	"runtime"
	"unsafe"
)

// ----------------------------------------------------------------------------
// Package initialization and availability
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Function type and lifecycle
// ----------------------------------------------------------------------------

var ErrInvalidFunctionId = errors.New("invalid function id")

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
		// NewFunction failures (missing source, MSL compile error, function not found) are not
		// invalid-handle conditions, so the code is errCodeNone and no sentinel is attached: the
		// handle does not exist yet.
		return nil, metalErrToError(err, "unable to set up metal function", errCodeNone)
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

// Close releases the compiled pipeline for this function. The Function becomes invalid after this
// call. It is the caller's responsibility to ensure no concurrent Run or String calls are in
// progress.
func (f *Function) Close() error {
	if f == nil || !f.Valid() {
		return ErrInvalidFunctionId
	}

	// The C side may strdup an error message into err on failure; we must free it. It also
	// categorizes the failure in code so metalErrToError can attach the matching sentinel.
	var err *C.char
	defer func() { freeCString(err) }()
	var code C.int

	if !C.function_close(C.int(f.id), &err, &code) {
		return metalErrToError(err, "unable to close metal function", code)
	}

	f.id = 0

	return nil
}

// ----------------------------------------------------------------------------
// Dispatch parameters
// ----------------------------------------------------------------------------

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

// A RunHandle represents an in-flight asynchronous dispatch started by RunAsync or RunBatchAsync.
// Call Wait exactly once to block until the GPU finishes and release the underlying command buffer.
// A RunHandle must not be used after Wait returns.
type RunHandle struct {
	handle unsafe.Pointer
}

// ----------------------------------------------------------------------------
// Dispatch: synchronous, batched, and asynchronous
// ----------------------------------------------------------------------------

// Run executes the computational function on the GPU. This can be called multiple times for the
// same Function Id and/or same buffers and is safe for concurrent use.
//
// A grid dimension of 0 is treated as 1 (so an unset dimension behaves as a single unit). A
// negative grid dimension is invalid and returns an error.
//
// On GPUs that support non-uniform threadgroup sizes (Apple4 and later, or the Mac2 family) the
// grid is dispatched exactly, so a kernel can assume its thread position is always within bounds.
// On older hardware the grid is rounded up to whole threadgroups, which over-dispatches: such a
// kernel must bounds-check its thread position against the real problem size before indexing a
// buffer.
func (f *Function) Run(params RunParameters) error {
	if err := Available(); err != nil {
		return err
	}

	width, height, depth, inputsPtr, bufferIdsPtr, err := params.prepare()
	if err != nil {
		return err
	}

	// The C side may strdup an error message into cErr on failure; we must free it. It also
	// categorizes the failure in code (invalid function id vs. invalid buffer id) so
	// metalErrToError can attach the matching sentinel.
	var cErr *C.char
	defer func() { freeCString(cErr) }()
	var code C.int

	// Run the computation on the GPU.
	ok := C.function_run(C.int(f.id), width, height, depth, inputsPtr, C.int(len(params.Inputs)), bufferIdsPtr, C.int(len(params.BufferIds)), &cErr, &code)

	// Keep the input and buffer-id slices alive until function_run returns. The C call reads through
	// inputsPtr/bufferIdsPtr (raw pointers into the slice backing arrays), which the Go garbage
	// collector cannot see; without these the collector would be free to reclaim the slices while
	// the GPU is still reading them.
	runtime.KeepAlive(params.Inputs)
	runtime.KeepAlive(params.BufferIds)

	if !ok {
		return metalErrToError(cErr, "unable to run metal function", code)
	}

	return nil
}

// RunBatch executes several dispatches of this function as a single GPU command buffer. Every
// element of params is dispatched in order against this same Function, then the whole batch is
// committed once and waited on once. Batching amortizes the per-command-buffer setup and the single
// CPU/GPU synchronization across all dispatches, so it is significantly faster than calling Run in a
// loop when you have many independent dispatches to run back to back.
//
// All dispatches share one command buffer, so they are not isolated: if any dispatch fails to encode
// (invalid buffer id, invalid grid), nothing is committed and RunBatch returns that dispatch's error.
// An empty params is a no-op that returns nil.
//
// Like Run, RunBatch is safe for concurrent use and blocks until the GPU finishes. The grid and
// over-dispatch semantics for each dispatch are identical to Run.
func (f *Function) RunBatch(params []RunParameters) error {
	if err := Available(); err != nil {
		return err
	}
	if len(params) == 0 {
		return nil
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()

	args, err := f.marshalBatch(params, &pinner)
	if err != nil {
		return err
	}

	var cErr *C.char
	defer func() { freeCString(cErr) }()
	var code C.int

	ok := C.function_run_batch(C.int(len(params)), &args.functionIds[0], &args.widths[0], &args.heights[0],
		&args.depths[0], &args.inputs[0], &args.numInputs[0], &args.bufferIds[0], &args.numBufferIds[0], &cErr, &code)

	if !ok {
		return metalErrToError(cErr, "unable to run metal function batch", code)
	}

	return nil
}

// RunAsync encodes and commits a dispatch like Run but returns immediately without waiting for the
// GPU to finish, so the caller can keep the CPU busy (encoding more work, doing CPU computation)
// while the GPU runs. The returned RunHandle must be passed to Wait exactly once to block for
// completion; the results in the output buffers are only valid after Wait returns.
//
// The buffers referenced by params.BufferIds must not be closed until Wait has returned, since the
// GPU reads and writes their shared memory while the dispatch is in flight. params.Inputs, by
// contrast, are copied during the call and need not outlive it.
//
// RunAsync is safe for concurrent use. The grid and over-dispatch semantics are identical to Run.
func (f *Function) RunAsync(params RunParameters) (*RunHandle, error) {
	if err := Available(); err != nil {
		return nil, err
	}

	width, height, depth, inputsPtr, bufferIdsPtr, err := params.prepare()
	if err != nil {
		return nil, err
	}

	var cErr *C.char
	defer func() { freeCString(cErr) }()
	var code C.int
	var handle unsafe.Pointer

	ok := C.function_run_async(C.int(f.id), width, height, depth, inputsPtr, C.int(len(params.Inputs)),
		bufferIdsPtr, C.int(len(params.BufferIds)), &handle, &cErr, &code)

	// Keep the slices alive through encoding (which happens synchronously inside the C call). After
	// the call returns the dispatch is fully encoded: inputs were copied via setBytes, and the
	// buffers are referenced from the command buffer and held alive by the buffer cache.
	runtime.KeepAlive(params.Inputs)
	runtime.KeepAlive(params.BufferIds)

	if !ok {
		return nil, metalErrToError(cErr, "unable to run metal function asynchronously", code)
	}

	return &RunHandle{handle: handle}, nil
}

// RunBatchAsync is the asynchronous counterpart of RunBatch: it encodes every dispatch into a single
// command buffer and commits it, but returns a RunHandle immediately instead of waiting. Because the
// whole batch is one command buffer, the single returned handle covers all of it — call Wait exactly
// once to block until every dispatch in the batch has finished.
//
// As with RunBatch, the dispatches are not isolated: if any one fails to encode, nothing is committed
// and RunBatchAsync returns that dispatch's error with a nil handle. An empty params returns a nil
// handle and nil error (there is nothing to wait on). As with RunAsync, the referenced buffers must
// not be closed until Wait has returned.
//
// RunBatchAsync is safe for concurrent use.
func (f *Function) RunBatchAsync(params []RunParameters) (*RunHandle, error) {
	if err := Available(); err != nil {
		return nil, err
	}
	if len(params) == 0 {
		return nil, nil
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()

	args, err := f.marshalBatch(params, &pinner)
	if err != nil {
		return nil, err
	}

	var cErr *C.char
	defer func() { freeCString(cErr) }()
	var code C.int
	var handle unsafe.Pointer

	ok := C.function_run_batch_async(C.int(len(params)), &args.functionIds[0], &args.widths[0], &args.heights[0],
		&args.depths[0], &args.inputs[0], &args.numInputs[0], &args.bufferIds[0], &args.numBufferIds[0], &handle, &cErr, &code)

	if !ok {
		return nil, metalErrToError(cErr, "unable to run metal function batch asynchronously", code)
	}

	return &RunHandle{handle: handle}, nil
}

// Wait blocks until the asynchronous work behind this handle finishes on the GPU and releases the
// underlying command buffer. It must be called exactly once per RunHandle returned by RunAsync or
// RunBatchAsync; calling it twice, or on a zero-value handle, returns an error rather than crashing.
// For a RunBatchAsync handle the single Wait covers the entire batch. After Wait returns, the output
// buffers hold the results.
func (h *RunHandle) Wait() error {
	if h == nil || h.handle == nil {
		return errors.New("invalid run handle")
	}

	var cErr *C.char
	defer func() { freeCString(cErr) }()

	ok := C.function_wait(h.handle, &cErr)

	// Clear the handle so a second Wait is a safe no-op error rather than a double free of the
	// command buffer (function_wait took ownership of the retain via __bridge_transfer).
	h.handle = nil

	if !ok {
		return metalErrToError(cErr, "unable to wait for metal function", errCodeNone)
	}

	return nil
}

// ----------------------------------------------------------------------------
// Internal dispatch helpers
// ----------------------------------------------------------------------------

// prepare validates the grid and returns the C-typed grid dimensions plus pointers into the Inputs
// and BufferIds backing arrays. float32/C.float and int32/C.int are binary compatible on all Apple
// platforms, so the slices are cast directly without copying. The returned pointers are only valid
// while the caller keeps params.Inputs and params.BufferIds alive (see runtime.KeepAlive at the
// call sites).
func (params RunParameters) prepare() (width, height, depth C.uint, inputsPtr *C.float, bufferIdsPtr *C.int, err error) {
	// Every dimension must be at least one unit long. A zero dimension is a convenience for "unused"
	// and clamps to 1; a negative dimension is a caller bug.
	if width, err = gridDimension(params.Grid.X); err != nil {
		return
	}
	if height, err = gridDimension(params.Grid.Y); err != nil {
		return
	}
	if depth, err = gridDimension(params.Grid.Z); err != nil {
		return
	}

	if len(params.Inputs) > 0 {
		inputsPtr = (*C.float)(unsafe.Pointer(&params.Inputs[0]))
	}
	if len(params.BufferIds) > 0 {
		bufferIdsPtr = (*C.int)(unsafe.Pointer(&params.BufferIds[0]))
	}

	return
}

// batchArgs holds the parallel C arrays that the batch entry points read, one element per dispatch.
type batchArgs struct {
	functionIds  []C.int
	widths       []C.uint
	heights      []C.uint
	depths       []C.uint
	inputs       []*C.float
	numInputs    []C.int
	bufferIds    []*C.int
	numBufferIds []C.int
}

// marshalBatch validates every dispatch and builds the parallel C arrays for the batch entry points.
//
// inputs[i] and bufferIds[i] are Go pointers into each RunParameters' slice backing arrays, and they
// are stored inside Go slices that are then passed to C by address. cgo forbids handing C a Go
// pointer that points at memory containing other (unpinned) Go pointers, so each inner pointer is
// pinned via pinner. Pinning also keeps the backing arrays alive for the call, so no separate
// runtime.KeepAlive is needed. The caller owns pinner and must Unpin it once the C call returns.
func (f *Function) marshalBatch(params []RunParameters, pinner *runtime.Pinner) (batchArgs, error) {
	n := len(params)
	args := batchArgs{
		functionIds:  make([]C.int, n),
		widths:       make([]C.uint, n),
		heights:      make([]C.uint, n),
		depths:       make([]C.uint, n),
		inputs:       make([]*C.float, n),
		numInputs:    make([]C.int, n),
		bufferIds:    make([]*C.int, n),
		numBufferIds: make([]C.int, n),
	}

	for i := range params {
		width, height, depth, inputsPtr, bufferIdsPtr, err := params[i].prepare()
		if err != nil {
			return batchArgs{}, err
		}

		args.functionIds[i] = C.int(f.id)
		args.widths[i] = width
		args.heights[i] = height
		args.depths[i] = depth
		if inputsPtr != nil {
			pinner.Pin(inputsPtr)
		}
		args.inputs[i] = inputsPtr
		args.numInputs[i] = C.int(len(params[i].Inputs))
		if bufferIdsPtr != nil {
			pinner.Pin(bufferIdsPtr)
		}
		args.bufferIds[i] = bufferIdsPtr
		args.numBufferIds[i] = C.int(len(params[i].BufferIds))
	}

	return args, nil
}

// gridDimension validates and normalizes a single grid dimension for the C layer. A size of 0 (an
// unused dimension) clamps to 1; a negative size is a caller error. The C side takes the dimensions
// as a 32-bit unsigned int, so a value above MaxInt32 is rejected rather than silently truncated
// when the Go int (up to 64-bit) is narrowed to C.uint.
func gridDimension(size int) (C.uint, error) {
	switch {
	case size < 0:
		return 0, errors.New("invalid grid dimension")
	case size > math.MaxInt32:
		return 0, errors.New("grid dimension exceeds maximum")
	case size == 0:
		return 1, nil
	default:
		return C.uint(size), nil
	}
}
