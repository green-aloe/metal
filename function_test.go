//go:build darwin

package metal

import (
	_ "embed"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

var (
	//go:embed test/noop.metal
	sourceNoop string
	//go:embed test/transfer1D.metal
	sourceTransfer1D string
	//go:embed test/transfer2D.metal
	sourceTransfer2D string
	//go:embed test/transfer3D.metal
	sourceTransfer3D string
	//go:embed test/sine.metal
	sourceSine string
	//go:embed test/transferType.metal
	sourceTransferType string
)

var (
	// nextFunctionId and nextBufferId track the IDs that should be returned for the next function
	// and buffer respectively. We use these to verify the caches are working correctly — each new
	// resource must get the next sequential ID from its own counter. Because these are global
	// variables, all tests that create new metal resources must be run serially (the default).
	nextFunctionId = 1
	nextBufferId   = 1
)

// validFunctionId tests that a Function ID has the expected value.
func validFunctionId(id int32) bool {
	ok := int(id) == nextFunctionId
	if ok {
		nextFunctionId++
	}
	return ok
}

// validBufferId tests that a BufferId has the expected value.
func validBufferId(id BufferId) bool {
	ok := int(id) == nextBufferId
	if ok {
		nextBufferId++
	}
	return ok
}

// addFunctionId marks that another function ID was returned.
func addFunctionId() {
	nextFunctionId++
}

// addBufferId marks that another buffer ID was returned.
func addBufferId() {
	nextBufferId++
}

// Test_Available tests that Available reports the package's initialization state and stays
// consistent with the public entry points.
func Test_Available(t *testing.T) {
	// These tests run on a machine with a working GPU, so initialization is expected to have
	// succeeded. Available must agree with the internal flag and return no error.
	require.True(t, metalAvailable)
	require.NoError(t, Available())

	// When Metal is available, NewFunction and NewBuffer must not short-circuit with
	// ErrMetalUnavailable; they should proceed to their normal success paths. We advance the global
	// id counters with addFunctionId/addBufferId (rather than asserting an exact next id) so this
	// test does not depend on running before or after the sequential-id tests.
	function, err := NewFunction(sourceNoop, "noop")
	require.NoError(t, err)
	require.True(t, function.Valid())
	addFunctionId()

	bufferId, _, err := NewBuffer[float32](1)
	require.NoError(t, err)
	require.True(t, bufferId.Valid())
	addBufferId()
}

// Test_Function_NewFunction tests that NewFunction either creates a new metal function or returns
// the expected message, depending on the conditions of each scenario.
func Test_Function_NewFunction(t *testing.T) {
	type subtest struct {
		name     string
		source   string
		function string
		wantErrs []string
	}

	subtests := []subtest{
		{
			name:     " no source or function name",
			wantErrs: []string{"unable to set up metal function: missing metal code"},
		},
		{
			name:     "invalid source, no function name",
			source:   "invalid",
			wantErrs: []string{"unable to set up metal function: missing function name"},
		},
		{
			name:     "no source, invalid function name",
			function: "invalid",
			wantErrs: []string{"unable to set up metal function: missing metal code"},
		},
		{
			name:     "invalid source, invalid function name",
			source:   "invalid",
			function: "invalid",
			wantErrs: []string{"unable to set up metal function: failed to create library", "unknown type name 'invalid'"},
		},
		{
			name:     "valid source, no function name",
			source:   sourceTransfer1D,
			function: "",
			wantErrs: []string{"unable to set up metal function: missing function name"},
		},
		{
			name:     "valid source, invalid function name",
			source:   sourceTransfer1D,
			function: "invalid",
			wantErrs: []string{"unable to set up metal function: failed to find function 'invalid'"},
		},
		{
			name:     "valid source, valid function name",
			source:   sourceTransfer1D,
			function: "transfer1D",
		},
	}

	for _, subtest := range subtests {
		t.Run(subtest.name, func(t *testing.T) {
			// Try to create a new metal function with the provided source and function name.
			function, err := NewFunction(subtest.source, subtest.function)

			// Test that the subtest's expected error and the actual error line up.
			switch len(subtest.wantErrs) {
			case 0:
				require.NoError(t, err, "Unable to create metal function: %s", err)
				require.True(t, function.Valid())
				require.True(t, validFunctionId(function.id))
			case 1:
				require.EqualError(t, err, subtest.wantErrs[0])
				require.False(t, function.Valid())
				require.False(t, validFunctionId(0))
			default:
				for _, wantErr := range subtest.wantErrs {
					require.ErrorContains(t, err, wantErr)
				}
				require.False(t, function.Valid())
				require.False(t, validFunctionId(0))
			}
		})
	}

	// Create a range of new functions and test that the returned function Id is always incremented
	// by 1.
	for i := 0; i < 100; i++ {
		function, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.NoError(t, err)
		require.True(t, validFunctionId(function.id), "%d %d %v %d", nextFunctionId, i, function, function.id)
	}
}

// Test_Function_Close tests that Function's Close method correctly releases the function.
func Test_Function_Close(t *testing.T) {
	t.Run("nil pointer", func(t *testing.T) {
		var nilPtr *Function
		require.ErrorIs(t, nilPtr.Close(), ErrInvalidFunctionId)
	})

	t.Run("uninitialized function", func(t *testing.T) {
		var function Function
		require.ErrorIs(t, function.Close(), ErrInvalidFunctionId)
	})

	t.Run("invalid cache id", func(t *testing.T) {
		function := Function{id: 99999}
		require.EqualError(t, function.Close(), "unable to close metal function: invalid function id: 99999")
	})

	t.Run("valid function", func(t *testing.T) {
		function, err := NewFunction(sourceNoop, "noop")
		require.NoError(t, err)
		require.True(t, validFunctionId(function.id))
		require.True(t, function.Valid())

		require.NoError(t, function.Close())
		require.False(t, function.Valid())
	})
}

// Test_Function_Valid tests that Function's Valid method correctly identifies a valid function Id.
func Test_Function_Valid(t *testing.T) {
	// A valid function Id has a positive value. Let's run through a bunch of numbers and test that
	// Valid always reports the correct status.
	for i := int32(-100_00); i <= 100_000; i++ {
		function := Function{id: i}

		if i > 0 {
			require.True(t, function.Valid())
		} else {
			require.False(t, function.Valid())
		}
	}
}

// Test_Function_String tests that Function's String method returns the correct function name.
func Test_Function_String(t *testing.T) {
	t.Run("uninitialized function", func(t *testing.T) {
		var function Function
		require.False(t, function.Valid())
		require.Equal(t, "", function.String())
	})

	t.Run("invalid function", func(t *testing.T) {
		function, err := NewFunction("", "")
		require.EqualError(t, err, "unable to set up metal function: missing metal code")
		require.False(t, function.Valid())
		require.Equal(t, "", function.String())
	})

	t.Run("valid function", func(t *testing.T) {
		function, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.NoError(t, err)
		require.True(t, function.Valid())
		require.True(t, validFunctionId(function.id))
		require.Equal(t, "transfer1D", function.String())
	})

	t.Run("closed function", func(t *testing.T) {
		// After Close, the function is no longer valid, so String reports the empty string rather than
		// reaching into the (now-released) cache. Close must be sequenced before String here; the two
		// are not safe to call concurrently on the same Function.
		function, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.NoError(t, err)
		require.True(t, validFunctionId(function.id))
		require.Equal(t, "transfer1D", function.String())

		require.NoError(t, function.Close())
		require.False(t, function.Valid())
		require.Equal(t, "", function.String())
	})
}

// Test_Function_NewFunction_threadSafe tests that NewFunction can handle multiple parallel invocations and
// still return the correct function Id.
func Test_Function_NewFunction_threadSafe(t *testing.T) {
	type data struct {
		function *Function
		wantName string
	}

	// We're going to use a wait group to block each goroutine after it's prepared until they're all
	// ready to fire.
	numIter := 100
	var wg sync.WaitGroup
	wg.Add(numIter)

	dataCh := make(chan data)

	// Prepare one goroutine to create a new function for each iteration.
	for i := 0; i < numIter; i++ {
		// Build the mock function name and mock metal code.
		functionName := fmt.Sprintf("abc_%d", i+1)
		source := fmt.Sprintf("kernel void %s() {}", functionName)

		// Spin up a new goroutine. This will wait until all goroutines are ready to fire, then
		// create a new metal function and send it back to the main thread.
		go func() {
			wg.Wait()

			function, err := NewFunction(source, functionName)
			require.NoError(t, err, "Unable to create metal function %s: %s", functionName, err)

			dataCh <- data{
				function: function,
				wantName: functionName,
			}
		}()

		// Mark that this goroutine is ready.
		wg.Done()
	}

	// Test that each function Id is unique and references the correct function.
	idMap := make(map[Function]struct{})
	for i := 0; i < numIter; i++ {
		data := <-dataCh

		_, ok := idMap[*data.function]
		require.False(t, ok)
		idMap[*data.function] = struct{}{}

		haveName := data.function.String()
		require.Equal(t, data.wantName, haveName)

		addFunctionId()
	}

	// Test that we received every Id in the sequence.
	idList := make([]Function, 0, len(idMap))
	for function := range idMap {
		idList = append(idList, function)
	}
	sort.Slice(idList, func(i, j int) bool { return idList[i].id < idList[j].id })
	require.Len(t, idList, numIter)
	for i := 0; i < numIter; i++ {
		require.Equal(t, nextFunctionId-numIter+i, int(idList[i].id))
	}
}

// Test_Function_Run_invalid tests that Function's Run method correctly handles invalid parameters.
func Test_Function_Run_invalid(t *testing.T) {
	function, err := NewFunction(sourceNoop, "noop")
	require.NoError(t, err)
	require.True(t, validFunctionId(function.id))

	t.Run("invalid (uninitialized) function", func(t *testing.T) {
		var emptyFunction Function
		err := emptyFunction.Run(RunParameters{})
		require.EqualError(t, err, "unable to run metal function: failed to retrieve function: invalid function id: 0")
		require.ErrorIs(t, err, ErrInvalidFunctionId)
	})

	t.Run("non-existent buffer", func(t *testing.T) {
		err := function.Run(RunParameters{BufferIds: []BufferId{10000}})
		require.EqualError(t, err, "unable to run metal function: failed to retrieve buffer 1/1: invalid buffer id: 10000")
		require.ErrorIs(t, err, ErrInvalidBufferId)
	})

	t.Run("negative grid X", func(t *testing.T) {
		err := function.Run(RunParameters{Grid: Grid{X: -1}})
		require.EqualError(t, err, "invalid grid dimension")
	})

	t.Run("negative grid Y", func(t *testing.T) {
		err := function.Run(RunParameters{Grid: Grid{Y: -1}})
		require.EqualError(t, err, "invalid grid dimension")
	})

	t.Run("negative grid Z", func(t *testing.T) {
		err := function.Run(RunParameters{Grid: Grid{Z: -1}})
		require.EqualError(t, err, "invalid grid dimension")
	})

	t.Run("oversized grid dimension", func(t *testing.T) {
		// A dimension above MaxInt32 would truncate when narrowed to the C layer's 32-bit unsigned
		// int, so each axis must reject it rather than silently dispatch a wrong-sized grid.
		for _, g := range []Grid{
			{X: math.MaxInt32 + 1},
			{Y: math.MaxInt32 + 1},
			{Z: math.MaxInt32 + 1},
		} {
			err := function.Run(RunParameters{Grid: g})
			require.EqualError(t, err, "grid dimension exceeds maximum")
		}
	})

	t.Run("zero grid clamps to one", func(t *testing.T) {
		// A fully-zero grid is the zero value of Grid; every dimension clamps to 1 and the noop
		// kernel (which takes no buffers) runs successfully.
		err := function.Run(RunParameters{Grid: Grid{X: 0, Y: 0, Z: 0}})
		require.NoError(t, err)
	})
}

// Test_Function_Run_1D tests that Function's Run method correctly runs a 1-dimensional
// computational process for small and large input sizes.
func Test_Function_Run_1D(t *testing.T) {
	for _, width := range []int{100, 100_000, 100_000_000} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			function, err := NewFunction(sourceTransfer1D, "transfer1D")
			require.NoError(t, err)
			require.True(t, validFunctionId(function.id))

			// Set up input and output buffers.
			inputId, input, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(inputId))
			outputId, output, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(outputId))

			// Set some initial values for the input.
			for i := range input {
				input[i] = float32(i + 1)
			}

			// Run the function and test that all values were transferred from the input to the output.
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
				},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, input, output)

			// Set some different values in the input and run the function again.
			for i := range input {
				input[i] = float32(i * i)
			}
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
				},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, input, output)
		})
	}
}

// Test_Function_Run_2D tests that Function's Run method correctly runs a 2-dimensional
// computational process for small and large input sizes.
func Test_Function_Run_2D(t *testing.T) {
	for _, width := range []int{10, 100, 10000} {
		height := width * 2

		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs and adds 1.
			function, err := NewFunction(sourceTransfer2D, "transfer2D")
			require.NoError(t, err)
			require.True(t, validFunctionId(function.id))

			// Set up input and output buffers.
			inputId, i, err := NewBuffer[float32](width * height)
			require.NoError(t, err)
			require.True(t, validBufferId(inputId))
			outputId, o, err := NewBuffer[float32](width * height)
			require.NoError(t, err)
			require.True(t, validBufferId(outputId))

			input := Fold(i, width)
			output := Fold(o, width)

			// Set some initial values for the input.
			for i := range input {
				for j := range input[i] {
					input[i][j] = float32(i*height + j)
				}
			}

			// Mirror the inputs to the expected outputs with an increment of 1.
			want := make([][]float32, len(input))
			for i := range input {
				want[i] = make([]float32, len(input[i]))
				for j := range input[i] {
					want[i][j] = input[i][j] + 1
				}
			}

			// Run the function and test that all values were transferred from the input to the output.
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
					Y: height,
				},
				Inputs:    []float32{1},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, want, output)

			// Set some different values in the input and run the function again.
			for i := range input {
				for j := range input[i] {
					input[i][j] = float32(i*height*2 + j + 100)
					want[i][j] = input[i][j] + 1
				}
			}
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
					Y: height,
				},
				Inputs:    []float32{1},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, want, output)
		})
	}
}

// Test_Function_Run_3D tests that Function's Run method correctly runs a 3-dimensional
// computational process for small and large input sizes.
func Test_Function_Run_3D(t *testing.T) {
	for _, width := range []int{10, 100, 250} {
		height := width * 2
		depth := width * 4

		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			function, err := NewFunction(sourceTransfer3D, "transfer3D")
			require.NoError(t, err)
			require.True(t, validFunctionId(function.id))

			// Set up input and output buffers.
			inputId, i, err := NewBuffer[float32](width * height * depth)
			require.NoError(t, err)
			require.True(t, validBufferId(inputId))
			outputId, o, err := NewBuffer[float32](width * height * depth)
			require.NoError(t, err)
			require.True(t, validBufferId(outputId))

			input := Fold(Fold(i, width*height), width)
			output := Fold(Fold(o, width*height), width)

			// Set some initial values for the input.
			for i := range input {
				for j := range input[i] {
					for k := range input[i][j] {
						input[i][j][k] = float32(i*height*depth + j*depth + k)
					}
				}
			}

			// Run the function and test that all values were transferred from the input to the output.
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
					Y: height,
					Z: depth,
				},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, input, output)

			// Set some different values in the input and run the function again.
			for i := range input {
				for j := range input[i] {
					for k := range input[i][j] {
						input[i][j][k] = float32(i*height*depth*2 + j*depth + k + 100)
					}
				}
			}
			require.NotEqual(t, input, output)
			err = function.Run(RunParameters{
				Grid: Grid{
					X: width,
					Y: height,
					Z: depth,
				},
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err)
			require.Equal(t, input, output)
		})
	}
}

// Test_Function_Run_threadSafe tests that Function's Run method can handle multiple parallel
// invocations and still operate on the correct set of buffers.
func Test_Function_Run_threadSafe(t *testing.T) {
	type data struct {
		iteration int
		input     []float32
		output    []float32
	}

	// Set up the metal function.
	function, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.NoError(t, err)
	require.True(t, validFunctionId(function.id))

	// We're going to use a wait group to block each goroutine after it's prepared until they're all
	// ready to fire.
	numIter := 100
	var wg sync.WaitGroup
	wg.Add(numIter)

	dataCh := make(chan data)

	width := 100_000
	grid := Grid{X: width}

	// Prepare one goroutine to run the metal function with unique buffers for each iteration.
	for iteration := 1; iteration <= numIter; iteration++ {
		// Create the buffers for this iteration.
		inputId, input, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, output, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		// Set values in the input so we can test that the output was operated on correctly.
		for i := range input {
			input[i] = float32(i * iteration)
		}

		// Spin up a new goroutine. This will wait until all goroutines are ready to fire, then
		// create a new metal function and send it back to the main thread.
		go func(iteration int) {
			wg.Wait()

			err := function.Run(RunParameters{
				Grid:      grid,
				BufferIds: []BufferId{inputId, outputId},
			})
			require.NoError(t, err, "Unable to run metal function (iteration %d): %v", iteration, err)

			dataCh <- data{
				iteration: iteration,
				input:     input,
				output:    output,
			}
		}(iteration)

		// Mark that this goroutine is ready.
		wg.Done()
	}

	// Test that each output received the correct values.
	for iteration := 1; iteration <= numIter; iteration++ {
		data := <-dataCh
		for i := range data.output {
			require.Equal(t, float32(i*data.iteration), data.output[i], "Iteration %d failed on item %d", data.iteration, i+1)
		}
	}
}

// Test_Function_types tests that specific primitive types in go line up with specific primitive
// types in metal.
func Test_Function_types(t *testing.T) {
	// The wantFail cases deliberately store a Go value into a buffer whose Metal type is narrower, so
	// the bytes are reinterpreted and the round-trip does not match. Since 64-bit Go types are no
	// longer part of BufferType, the "store a wider type" mismatches are demonstrated with the
	// narrower in-range types (float32 into half, int32 into short, uint32 into ushort).
	testType[float32](t, "float", false, func(i int) float32 { return float32(i) * 1.1 })

	// Go doesn't currently have an equivalent "float16" type
	testType[float32](t, "half", true, func(i int) float32 { return float32(i) * 1.1 })

	testType[int32](t, "int", false, func(i int) int32 { return int32(-i) })

	testType[int16](t, "short", false, func(i int) int16 { return int16(-i) })
	testType[int32](t, "short", true, func(i int) int32 { return int32(-i) })

	testType[uint32](t, "uint", false, func(i int) uint32 { return uint32(i) })

	testType[uint16](t, "ushort", false, func(i int) uint16 { return uint16(i) })
	testType[uint32](t, "ushort", true, func(i int) uint32 { return uint32(i) })
}

// testType runs a test for a buffer type.
func testType[T BufferType](t *testing.T, metalType string, wantFail bool, setter func(int) T) {
	t.Run(fmt.Sprintf("%s_%v", metalType, wantFail), func(t *testing.T) {
		// Build the metal code.
		source := fmt.Sprintf(sourceTransferType, metalType, metalType)

		// Set up a metal function.
		function, err := NewFunction(source, "transferType")
		require.NoError(t, err)
		require.True(t, validFunctionId(function.id))

		// Create the input and output buffers.
		inputId, input, err := NewBuffer[T](100)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, output, err := NewBuffer[T](100)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		// Set the inputs.
		for i := range input {
			input[i] = setter(i)
		}

		// Run the metal function.
		err = function.Run(RunParameters{
			Grid: Grid{
				X: 100,
			},
			BufferIds: []BufferId{inputId, outputId},
		})
		require.NoError(t, err)

		// Test that the inputs were either correctly or incorrectly transferred over to the
		// outputs, depending on the test scenario.
		if wantFail {
			require.NotEqual(t, input, output)
		} else {
			require.Equal(t, input, output)
		}
	})
}

// Test_Function_RunBatch tests that RunBatch dispatches every set of parameters against the same
// function in a single command buffer and that each dispatch operates on its own buffers.
func Test_Function_RunBatch(t *testing.T) {
	function, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.NoError(t, err)
	require.True(t, validFunctionId(function.id))

	t.Run("empty batch is a no-op", func(t *testing.T) {
		require.NoError(t, function.RunBatch(nil))
		require.NoError(t, function.RunBatch([]RunParameters{}))
	})

	t.Run("multiple dispatches", func(t *testing.T) {
		width := 1000
		numDispatches := 8

		// Each dispatch gets its own input/output buffer pair with distinct values, so a correct
		// batch must transfer each input into its matching output independently.
		params := make([]RunParameters, numDispatches)
		inputs := make([][]float32, numDispatches)
		outputs := make([][]float32, numDispatches)
		for d := range params {
			inputId, input, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(inputId))
			outputId, output, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(outputId))

			for i := range input {
				input[i] = float32(i*(d+1)) + 0.5
			}

			params[d] = RunParameters{Grid: Grid{X: width}, BufferIds: []BufferId{inputId, outputId}}
			inputs[d] = input
			outputs[d] = output
		}

		require.NoError(t, function.RunBatch(params))

		for d := range params {
			require.Equal(t, inputs[d], outputs[d], "dispatch %d did not transfer correctly", d)
		}
	})

	t.Run("invalid buffer in one dispatch fails the batch", func(t *testing.T) {
		width := 10
		inputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		err = function.RunBatch([]RunParameters{
			{Grid: Grid{X: width}, BufferIds: []BufferId{inputId, outputId}},
			{Grid: Grid{X: width}, BufferIds: []BufferId{10000, outputId}},
		})
		require.ErrorIs(t, err, ErrInvalidBufferId)
	})

	t.Run("invalid grid in one dispatch fails the batch", func(t *testing.T) {
		err := function.RunBatch([]RunParameters{
			{Grid: Grid{X: 10}},
			{Grid: Grid{X: -1}},
		})
		require.EqualError(t, err, "invalid grid dimension")
	})
}

// Test_Function_RunAsync tests that RunAsync starts a dispatch that produces correct results once
// Wait returns, and that the handle guards against misuse.
func Test_Function_RunAsync(t *testing.T) {
	function, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.NoError(t, err)
	require.True(t, validFunctionId(function.id))

	t.Run("results are correct after Wait", func(t *testing.T) {
		width := 100_000
		inputId, input, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, output, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		for i := range input {
			input[i] = float32(i) * 1.5
		}

		handle, err := function.RunAsync(RunParameters{
			Grid:      Grid{X: width},
			BufferIds: []BufferId{inputId, outputId},
		})
		require.NoError(t, err)
		require.NotNil(t, handle)

		require.NoError(t, handle.Wait())
		require.Equal(t, input, output)
	})

	t.Run("second Wait is a safe error", func(t *testing.T) {
		width := 10
		inputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		handle, err := function.RunAsync(RunParameters{
			Grid:      Grid{X: width},
			BufferIds: []BufferId{inputId, outputId},
		})
		require.NoError(t, err)
		require.NoError(t, handle.Wait())
		require.EqualError(t, handle.Wait(), "invalid run handle")
	})

	t.Run("nil handle Wait is a safe error", func(t *testing.T) {
		var handle *RunHandle
		require.EqualError(t, handle.Wait(), "invalid run handle")
	})

	t.Run("invalid buffer fails before returning a handle", func(t *testing.T) {
		handle, err := function.RunAsync(RunParameters{
			Grid:      Grid{X: 10},
			BufferIds: []BufferId{10000},
		})
		require.ErrorIs(t, err, ErrInvalidBufferId)
		require.Nil(t, handle)
	})

	t.Run("invalid grid fails before returning a handle", func(t *testing.T) {
		handle, err := function.RunAsync(RunParameters{Grid: Grid{X: -1}})
		require.EqualError(t, err, "invalid grid dimension")
		require.Nil(t, handle)
	})
}

// Test_Function_RunBatchAsync tests that RunBatchAsync commits a whole batch as one command buffer
// and that a single Wait completes every dispatch with correct results.
func Test_Function_RunBatchAsync(t *testing.T) {
	function, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.NoError(t, err)
	require.True(t, validFunctionId(function.id))

	t.Run("empty batch returns nil handle and nil error", func(t *testing.T) {
		handle, err := function.RunBatchAsync(nil)
		require.NoError(t, err)
		require.Nil(t, handle)
	})

	t.Run("results are correct after one Wait", func(t *testing.T) {
		width := 5000
		numDispatches := 6

		params := make([]RunParameters, numDispatches)
		inputs := make([][]float32, numDispatches)
		outputs := make([][]float32, numDispatches)
		for d := range params {
			inputId, input, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(inputId))
			outputId, output, err := NewBuffer[float32](width)
			require.NoError(t, err)
			require.True(t, validBufferId(outputId))

			for i := range input {
				input[i] = float32(i*(d+2)) - 0.25
			}

			params[d] = RunParameters{Grid: Grid{X: width}, BufferIds: []BufferId{inputId, outputId}}
			inputs[d] = input
			outputs[d] = output
		}

		handle, err := function.RunBatchAsync(params)
		require.NoError(t, err)
		require.NotNil(t, handle)

		require.NoError(t, handle.Wait())

		for d := range params {
			require.Equal(t, inputs[d], outputs[d], "dispatch %d did not transfer correctly", d)
		}

		// The batch is one command buffer, so a second Wait must be a safe error.
		require.EqualError(t, handle.Wait(), "invalid run handle")
	})

	t.Run("invalid buffer fails before returning a handle", func(t *testing.T) {
		width := 10
		inputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(inputId))
		outputId, _, err := NewBuffer[float32](width)
		require.NoError(t, err)
		require.True(t, validBufferId(outputId))

		handle, err := function.RunBatchAsync([]RunParameters{
			{Grid: Grid{X: width}, BufferIds: []BufferId{inputId, outputId}},
			{Grid: Grid{X: width}, BufferIds: []BufferId{10000, outputId}},
		})
		require.ErrorIs(t, err, ErrInvalidBufferId)
		require.Nil(t, handle)
	})

	t.Run("invalid grid fails before returning a handle", func(t *testing.T) {
		handle, err := function.RunBatchAsync([]RunParameters{
			{Grid: Grid{X: 10}},
			{Grid: Grid{X: -1}},
		})
		require.EqualError(t, err, "invalid grid dimension")
		require.Nil(t, handle)
	})
}

// Benchmark_Run benchmarks running a computational process for a wide range of widths both in the
// standard, serial method and in the GPU-accelerated parallel method.
func Benchmark_Run(b *testing.B) {
	for _, width := range []int{100, 100_000, 100_000_000} {

		// Set up a metal function.
		function, _ := NewFunction(sourceSine, "sine")
		addFunctionId()

		// Set up input and output buffers.
		inputId, input, _ := NewBuffer[float32](width)
		addBufferId()
		outputId, output, _ := NewBuffer[float32](width)
		addBufferId()

		for i := range input {
			input[i] = rand.Float32()
		}

		b.Run(fmt.Sprintf("Serial_%d", width), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				for i := range input {
					output[i] = float32(math.Sin(float64(input[i]))) * 0.01 * 0.01
				}
			}
		})

		b.Run(fmt.Sprintf("Parallel_%d", width), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				function.Run(RunParameters{
					Grid: Grid{
						X: width,
					},
					Inputs:    []float32{0.01},
					BufferIds: []BufferId{inputId, outputId},
				})
			}
		})
	}
}
