//go:build darwin
// +build darwin

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
	// nextMetalId tracks the Id that should be returned for the next metal resource. We're going to
	// use this to make sure the metal cache is working as expected. Every time a new metal function
	// or metal buffer is created, this should be incremented. Because this is a global variable,
	// all tests that create new metal resources must be run concurrently.
	nextMetalId = 1
)

// validId tests that the Id has the expected value.
func validId[T int | BufferId](id T) bool {
	ok := int(id) == nextMetalId
	if ok {
		addId()
	}

	return ok
}

// addId marks that another Id was returned for a metal resource.
func addId() {
	nextMetalId++
}

// Test_Function_NewFunction tests that NewFunction either creates a new metal function or returns
// the expected message, depending on the conditions of each scenario.
func Test_Function_NewFunction(t *testing.T) {
	type subtest struct {
		name     string
		source   string
		function string
		wantErr  string
	}

	subtests := []subtest{
		{
			name:    " no source or function name",
			wantErr: "Unable to set up metal function: Missing metal code",
		},
		{
			name:    "invalid source, no function name",
			source:  "invalid",
			wantErr: "Unable to set up metal function: Missing function name",
		},
		{
			name:     "no source, invalid function name",
			function: "invalid",
			wantErr:  "Unable to set up metal function: Missing metal code",
		},
		{
			name:     "invalid source, invalid function name",
			source:   "invalid",
			function: "invalid",
			wantErr:  "Unable to set up metal function: Failed to create library (see console log)",
		},
		{
			name:     "valid source, no function name",
			source:   sourceTransfer1D,
			function: "",
			wantErr:  "Unable to set up metal function: Missing function name",
		},
		{
			name:     "valid source, invalid function name",
			source:   sourceTransfer1D,
			function: "invalid",
			wantErr:  "Unable to set up metal function: Failed to find function 'invalid'",
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
			if subtest.wantErr == "" {
				require.Nil(t, err, "Unable to create metal function: %s", err)
				require.True(t, function.Valid())
				require.True(t, validId(function.id))
			} else {
				require.NotNil(t, err)
				require.Equal(t, subtest.wantErr, err.Error())
				require.False(t, function.Valid())
			}
		})
	}

	// Create a range of new functions and test that the returned function Id is always incremented
	// by 1.
	for i := 0; i < 100; i++ {
		function, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.Nil(t, err)
		require.True(t, validId(function.id))
	}
}

// Test_Function_Valid tests that Function's Valid method correctly identifies a valid function Id.
func Test_Function_Valid(t *testing.T) {
	// A valid function Id has a positive value. Let's run through a bunch of numbers and test that
	// Valid always reports the correct status.
	for i := -100_00; i <= 100_000; i++ {
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
		require.Equal(t, "", function.String())
	})

	t.Run("invalid function", func(t *testing.T) {
		function, err := NewFunction("", "")
		require.NotNil(t, err)
		require.False(t, function.Valid())
		require.Equal(t, "", function.String())
	})

	t.Run("valid function", func(t *testing.T) {
		function, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.Nil(t, err)
		require.True(t, function.Valid())
		require.True(t, validId(function.id))
		require.Equal(t, "transfer1D", function.String())
	})
}

// Test_Function_NewFunction_threadSafe tests that NewFunction can handle multiple parallel invocations and
// still return the correct function Id.
func Test_Function_NewFunction_threadSafe(t *testing.T) {
	type data struct {
		function Function
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
			require.Nil(t, err, "Unable to create metal function %s: %s", functionName, err)

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

		_, ok := idMap[data.function]
		require.False(t, ok)
		idMap[data.function] = struct{}{}

		haveName := data.function.String()
		require.Equal(t, data.wantName, haveName)

		addId()
	}

	// Test that we received every Id in the sequence.
	idList := make([]Function, 0, len(idMap))
	for function := range idMap {
		idList = append(idList, function)
	}
	sort.Slice(idList, func(i, j int) bool { return idList[i].id < idList[j].id })
	require.Len(t, idList, numIter)
	for i := 0; i < numIter; i++ {
		require.Equal(t, nextMetalId-numIter+i, int(idList[i].id))
	}
}

// Test_Function_Run_invalid tests that Function's Run method correctly handles invalid parameters.
func Test_Function_Run_invalid(t *testing.T) {
	function, err := NewFunction(sourceNoop, "noop")
	require.Nil(t, err)
	require.True(t, validId(function.id))

	t.Run("invalid (uninitialized) function", func(t *testing.T) {
		var emptyFunction Function
		err := emptyFunction.Run(RunParameters{})
		require.NotNil(t, err)
		require.Equal(t, "Unable to run metal function: Failed to retrieve function", err.Error())
	})

	t.Run("non-existent buffer", func(t *testing.T) {
		err := function.Run(RunParameters{BufferIds: []BufferId{10000}})
		require.NotNil(t, err)
		require.Equal(t, "Unable to run metal function: Failed to retrieve buffer 1/1 using Id 10000", err.Error())
	})

	t.Run("invalid grid", func(t *testing.T) {
		err := function.Run(RunParameters{Grid: Grid{X: -1, Y: -1, Z: -1}})
		require.Nil(t, err)
	})
}

// Test_Function_Run_1D tests that Function's Run method correctly runs a 1-dimensional
// computational process for small and large input sizes.
func Test_Function_Run_1D(t *testing.T) {
	for _, width := range []int{100, 100_000, 100_000_000} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			function, err := NewFunction(sourceTransfer1D, "transfer1D")
			require.Nil(t, err)
			require.True(t, validId(function.id))

			// Set up input and output buffers.
			inputId, input, err := NewBuffer[float32](width)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, output, err := NewBuffer[float32](width)
			require.Nil(t, err)
			require.True(t, validId(outputId))

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
			require.Nil(t, err)
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
			require.Nil(t, err)
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
			require.Nil(t, err)
			require.True(t, validId(function.id))

			// Set up input and output buffers.
			inputId, i, err := NewBuffer[float32](width * height)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, o, err := NewBuffer[float32](width * height)
			require.Nil(t, err)
			require.True(t, validId(outputId))

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
			require.Nil(t, err)
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
			require.Nil(t, err)
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
			require.Nil(t, err)
			require.True(t, validId(function.id))

			// Set up input and output buffers.
			inputId, i, err := NewBuffer[float32](width * height * depth)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, o, err := NewBuffer[float32](width * height * depth)
			require.Nil(t, err)
			require.True(t, validId(outputId))

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
			require.Nil(t, err)
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
			require.Nil(t, err)
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
		err       error
	}

	// Set up the metal function.
	function, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.Nil(t, err)
	require.True(t, validId(function.id))

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
		require.Nil(t, err)
		require.True(t, validId(inputId))
		outputId, output, err := NewBuffer[float32](width)
		require.Nil(t, err)
		require.True(t, validId(outputId))

		// Set values in the input so we can test that the output was operated on correctly.
		for i := range input {
			input[i] = float32(i * iteration)
		}

		// Spin up a new goroutine. This will wait until all goroutines are ready to fire, then
		// create a new metal function and send it back to the main thread.
		go func(iteration int) {
			wg.Wait()

			err := function.Run(grid, inputId, outputId)

			dataCh <- data{
				iteration: iteration,
				input:     input,
				output:    output,
				err:       err,
			}
		}(iteration)

		// Mark that this goroutine is ready.
		wg.Done()
	}

	// Test that each output received the correct values.
	for iteration := 1; iteration <= numIter; iteration++ {
		data := <-dataCh
		require.Nil(t, err, "Unable to run metal function (iteration %d): %s", data.iteration, err)

		for i := range data.output {
			require.Equal(t, float32(i*data.iteration), data.output[i], "Iteration %d failed on item %d", data.iteration, i+1)
		}
	}
}

// Test_Function_types tests that specific primitive types in go line up with specific primitive
// types in metal.
func Test_Function_types(t *testing.T) {
	testType[float32](t, "float", false, func(i int) float32 { return float32(i) * 1.1 })
	testType[float64](t, "float", true, func(i int) float64 { return float64(i) * 1.1 })

	// Go doesn't currently have an equivalent "float16" type
	testType[float32](t, "half", true, func(i int) float32 { return float32(i) * 1.1 })

	testType[int32](t, "int", false, func(i int) int32 { return int32(-i) })
	testType[int64](t, "int", true, func(i int) int64 { return int64(-i) })

	testType[int16](t, "short", false, func(i int) int16 { return int16(-i) })
	testType[int32](t, "short", true, func(i int) int32 { return int32(-i) })

	testType[uint32](t, "uint", false, func(i int) uint32 { return uint32(i) })
	testType[uint64](t, "uint", true, func(i int) uint64 { return uint64(i) })

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
		require.Nil(t, err)
		require.True(t, validId(function.id))

		// Create the input and output buffers.
		inputId, input, err := NewBuffer[T](100)
		require.Nil(t, err)
		require.True(t, validId(inputId))
		outputId, output, err := NewBuffer[T](100)
		require.Nil(t, err)
		require.True(t, validId(outputId))

		// Set the inputs.
		for i := range input {
			input[i] = setter(i)
		}

		// Run the metal function.
		function.Run(Grid{X: 100}, inputId, outputId)
		require.Nil(t, err)

		// Test that the inputs were either correctly or incorrectly transferred over to the
		// outputs, depending on the test scenario.
		if wantFail {
			require.NotEqual(t, input, output)
		} else {
			require.Equal(t, input, output)
		}
	})
}

// Benchmark_Run benchmarks running a computational process for a wide range of widths both in the
// standard, serial method and in the GPU-accelerated parallel method.
func Benchmark_Run(b *testing.B) {
	for _, width := range []int{100, 100_000, 100_000_000} {

		// Set up a metal function.
		function, _ := NewFunction(sourceSine, "sine")
		addId()

		// Set up input and output buffers.
		inputId, input, _ := NewBuffer[float32](width)
		addId()
		outputId, output, _ := NewBuffer[float32](width)
		addId()

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
				function.Run(Grid{X: 1}, inputId, outputId)
			}
		})
	}
}
