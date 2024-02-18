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
func validId[T FunctionId | BufferId](id T) bool {
	ok := int(id) == nextMetalId
	if ok {
		nextMetalId++
	}

	return ok
}

// addId marks that another Id was returned for a metal resource.
func addId() {
	nextMetalId++
}

// Test_FunctionId_NewFunction tests that NewFunction either creates a new metal function or returns
// the expected message, depending on the conditions of each scenario.
func Test_FunctionId_NewFunction(t *testing.T) {
	type scenario struct {
		source   string
		function string
		wantErr  string
	}

	scenarios := []scenario{
		{
			// No source or function name
			wantErr: "Unable to set up metal function: Missing metal code",
		},
		{
			// Invalid source, no function name
			source:  "invalid",
			wantErr: "Unable to set up metal function: Missing function name",
		},
		{
			// No source, invalid function name
			function: "invalid",
			wantErr:  "Unable to set up metal function: Missing metal code",
		},
		{
			// Invalid source, invalid function name
			source:   "invalid",
			function: "invalid",
			wantErr:  "Unable to set up metal function: Failed to create library (see console log)",
		},
		{
			// Valid source, no function name
			source:   sourceTransfer1D,
			function: "",
			wantErr:  "Unable to set up metal function: Missing function name",
		},
		{
			// Valid source, invalid function name
			source:   sourceTransfer1D,
			function: "invalid",
			wantErr:  "Unable to set up metal function: Failed to find function 'invalid'",
		},
		{
			// Valid source, valid function name
			source:   sourceTransfer1D,
			function: "transfer1D",
		},
	}

	for i, scenario := range scenarios {
		t.Run(fmt.Sprintf("Scenario%02d", i+1), func(t *testing.T) {
			// Try to create a new metal function with the provided source and function name.
			functionId, err := NewFunction(scenario.source, scenario.function)

			// Test that the scenario's expected error and the actual error line up.
			if scenario.wantErr == "" {
				require.Nil(t, err, "Unable to create metal function: %s", err)
				require.True(t, functionId.Valid())
				require.True(t, validId(functionId))
			} else {
				require.NotNil(t, err)
				require.Equal(t, scenario.wantErr, err.Error())
				require.False(t, functionId.Valid())
			}
		})
	}

	// Create a range of new functions and test that the returned function Id is always incremented
	// by 1.
	for i := 0; i < 100; i++ {
		functionId, err := NewFunction(sourceTransfer1D, "transfer1D")
		require.Nil(t, err)
		require.True(t, validId(functionId))
	}
}

// Test_FunctionId_Valid tests that FunctionId's Valid method correctly identifies a valid function Id.
func Test_FunctionId_Valid(t *testing.T) {
	// A valid function Id has a positive value. Let's run through a bunch of numbers and test that
	// Valid always reports the correct status.
	for i := -100_00; i <= 100_000; i++ {
		function := FunctionId(i)

		if i > 0 {
			require.True(t, function.Valid())
		} else {
			require.False(t, function.Valid())
		}
	}
}

// Test_FunctionId_String tests that FunctionId's String method returns the correct function name.
func Test_FunctionId_String(t *testing.T) {
	// Test an uninitialized function.
	var function FunctionId
	require.Equal(t, "", function.String())

	// Test an invalid function.
	functionId, err := NewFunction("", "")
	require.NotNil(t, err)
	require.False(t, functionId.Valid())
	require.Equal(t, "", functionId.String())

	// Test a valid function.
	functionId, err = NewFunction(sourceTransfer1D, "transfer1D")
	require.Nil(t, err)
	require.True(t, functionId.Valid())
	require.True(t, validId(functionId))
	require.Equal(t, "transfer1D", functionId.String())
}

// Test_FunctionId_NewFunction_threadSafe tests that NewFunction can handle multiple parallel invocations and
// still return the correct function Id.
func Test_FunctionId_NewFunction_threadSafe(t *testing.T) {
	type data struct {
		functionId FunctionId
		wantName   string
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

			functionId, err := NewFunction(source, functionName)
			require.Nil(t, err, "Unable to create metal function %s: %s", functionName, err)

			dataCh <- data{
				functionId: functionId,
				wantName:   functionName,
			}
		}()

		// Mark that this goroutine is ready.
		wg.Done()
	}

	// Test that each function Id is unique and references the correct function.
	idMap := make(map[FunctionId]struct{})
	for i := 0; i < numIter; i++ {
		data := <-dataCh

		_, ok := idMap[data.functionId]
		require.False(t, ok)
		idMap[data.functionId] = struct{}{}

		haveName := data.functionId.String()
		require.Equal(t, data.wantName, haveName)

		addId()
	}

	// Test that we received every Id in the sequence.
	idList := make([]FunctionId, 0, len(idMap))
	for functionId := range idMap {
		idList = append(idList, functionId)
	}
	sort.Slice(idList, func(i, j int) bool { return idList[i] < idList[j] })
	require.Len(t, idList, numIter)
	for i := 0; i < numIter; i++ {
		require.Equal(t, nextMetalId-numIter+i, int(idList[i]))
	}
}

// Test_FunctionId_Run_invalid tests that FunctionId's Run method correctly handles invalid
// parameters.
func Test_FunctionId_Run_invalid(t *testing.T) {
	functionId, err := NewFunction(sourceNoop, "noop")
	require.Nil(t, err)
	require.True(t, validId(functionId))

	// Test calling Run with an invalid (uninitialized) Function.
	var emptyFunction FunctionId
	err = emptyFunction.Run(Grid{})
	require.NotNil(t, err)
	require.Equal(t, "Unable to run metal function: Failed to retrieve function", err.Error())

	// Test calling Run with a buffer Id for a buffer that doesn't exist.
	err = functionId.Run(Grid{}, BufferId(10000))
	require.NotNil(t, err)
	require.Equal(t, "Unable to run metal function: Failed to retrieve buffer 1/1 using Id 10000", err.Error())

	// Test calling Run with an invalid Grid.
	err = functionId.Run(Grid{X: -1, Y: -1, Z: -1})
	require.Nil(t, err)
}

// Test_FunctionId_Run_1D tests that FunctionId's Run method correctly runs a 1-dimensional
// computational process for small and large input sizes.
func Test_FunctionId_Run_1D(t *testing.T) {
	for _, width := range []int{100, 100_000, 100_000_000} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			functionId, err := NewFunction(sourceTransfer1D, "transfer1D")
			require.Nil(t, err)
			require.True(t, validId(functionId))

			// Set up input and output buffers.
			inputId, input, err := NewBuffer1D[float32](width)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, output, err := NewBuffer1D[float32](width)
			require.Nil(t, err)
			require.True(t, validId(outputId))

			// Set some initial values for the input.
			for i := range input {
				input[i] = float32(i + 1)
			}

			// Run the function and test that all values were transferred from the input to the output.
			require.NotEqual(t, input, output)
			err = functionId.Run(Grid{X: width}, inputId, outputId)
			require.Nil(t, err)
			require.Equal(t, input, output)

			// Set some different values in the input and run the function again.
			for i := range input {
				input[i] = float32(i * i)
			}
			require.NotEqual(t, input, output)
			err = functionId.Run(Grid{X: width}, inputId, outputId)
			require.Nil(t, err)
			require.Equal(t, input, output)
		})
	}
}

// Test_FunctionId_Run_2D tests that FunctionId's Run method correctly runs a 2-dimensional
// computational process for small and large input sizes.
func Test_FunctionId_Run_2D(t *testing.T) {
	for _, width := range []int{10, 100, 10000} {
		height := width * 2

		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			functionId, err := NewFunction(sourceTransfer2D, "transfer2D")
			require.Nil(t, err)
			require.True(t, validId(functionId))

			// Set up input and output buffers.
			inputId, input, err := NewBuffer2D[float32](width, height)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, output, err := NewBuffer2D[float32](width, height)
			require.Nil(t, err)
			require.True(t, validId(outputId))

			// Set some initial values for the input.
			for i := range input {
				for j := range input[i] {
					input[i][j] = float32(i*height + j)
				}
			}

			// Run the function and test that all values were transferred from the input to the output.
			require.NotEqual(t, input, output)
			err = functionId.Run(Grid{X: width, Y: height}, inputId, outputId)
			require.Nil(t, err)
			require.Equal(t, input, output)

			// Set some different values in the input and run the function again.
			for i := range input {
				for j := range input[i] {
					input[i][j] = float32(i*height*2 + j + 100)
				}
			}
			require.NotEqual(t, input, output)
			err = functionId.Run(Grid{X: width, Y: height}, inputId, outputId)
			require.Nil(t, err)
			require.Equal(t, input, output)
		})
	}
}

// Test_FunctionId_Run_3D tests that FunctionId's Run method correctly runs a 3-dimensional
// computational process for small and large input sizes.
func Test_FunctionId_Run_3D(t *testing.T) {
	for _, width := range []int{10, 100, 250} {
		height := width * 2
		depth := width * 4

		t.Run(strconv.Itoa(width), func(t *testing.T) {

			// Set up a metal function that simply transfers all inputs to the outputs.
			functionId, err := NewFunction(sourceTransfer3D, "transfer3D")
			require.Nil(t, err)
			require.True(t, validId(functionId))

			// Set up input and output buffers.
			inputId, input, err := NewBuffer3D[float32](width, height, depth)
			require.Nil(t, err)
			require.True(t, validId(inputId))
			outputId, output, err := NewBuffer3D[float32](width, height, depth)
			require.Nil(t, err)
			require.True(t, validId(outputId))

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
			err = functionId.Run(Grid{X: width, Y: height, Z: depth}, inputId, outputId)
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
			err = functionId.Run(Grid{X: width, Y: height, Z: depth}, inputId, outputId)
			require.Nil(t, err)
			require.Equal(t, input, output)
		})
	}
}

// Test_FunctionId_Run_threadSafe tests that FunctionId's Run method can handle multiple parallel
// invocations and still operate on the correct set of buffers.
func Test_FunctionId_Run_threadSafe(t *testing.T) {
	type data struct {
		iteration int
		input     []float32
		output    []float32
		err       error
	}

	// Set up the metal function.
	functionId, err := NewFunction(sourceTransfer1D, "transfer1D")
	require.Nil(t, err)
	require.True(t, validId(functionId))

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
		inputId, input, err := NewBuffer1D[float32](width)
		require.Nil(t, err)
		require.True(t, validId(inputId))
		outputId, output, err := NewBuffer1D[float32](width)
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

			err := functionId.Run(grid, inputId, outputId)

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

// Test_FunctionId_types tests that specific primitive types in go line up with specific primitive
// types in metal.
func Test_FunctionId_types(t *testing.T) {
	for _, metalType := range []string{
		"float",
		"half",
		"int",
		"short",
		"uint",
		"ushort",
	} {
		switch metalType {
		case "float":
			testType[float32](t, metalType, false, func(i int) float32 { return float32(i) * 1.1 })
			testType[float64](t, metalType, true, func(i int) float64 { return float64(i) * 1.1 })
		case "half":
			// Go doesn't currently have an equivalent "float16" type
			testType[float32](t, metalType, true, func(i int) float32 { return float32(i) * 1.1 })
		case "int":
			testType[int32](t, metalType, false, func(i int) int32 { return int32(-i) })
			testType[int64](t, metalType, true, func(i int) int64 { return int64(-i) })
		case "short":
			testType[int16](t, metalType, false, func(i int) int16 { return int16(-i) })
			testType[int32](t, metalType, true, func(i int) int32 { return int32(-i) })
		case "uint":
			testType[uint32](t, metalType, false, func(i int) uint32 { return uint32(i) })
			testType[uint64](t, metalType, true, func(i int) uint64 { return uint64(i) })
		case "ushort":
			testType[uint16](t, metalType, false, func(i int) uint16 { return uint16(i) })
			testType[uint32](t, metalType, true, func(i int) uint32 { return uint32(i) })
		}
	}
}

// testType runs a test for a buffer type.
func testType[T BufferType](t *testing.T, metalType string, wantFail bool, setter func(int) T) {
	t.Run(fmt.Sprintf("%s_%v", metalType, wantFail), func(t *testing.T) {
		// Build the metal code.
		source := fmt.Sprintf(sourceTransferType, metalType, metalType)

		// Set up a metal function.
		functionId, err := NewFunction(source, "transferType")
		require.Nil(t, err)
		require.True(t, validId(functionId))

		// Create the input and output buffers.
		inputId, input, err := NewBuffer1D[T](100)
		require.Nil(t, err)
		require.True(t, validId(inputId))
		outputId, output, err := NewBuffer1D[T](100)
		require.Nil(t, err)
		require.True(t, validId(outputId))

		// Set the inputs.
		for i := range input {
			input[i] = setter(i)
		}

		// Run the metal function.
		functionId.Run(Grid{X: 100}, inputId, outputId)
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
		functionId, _ := NewFunction(sourceSine, "sine")
		addId()

		// Set up input and output buffers.
		inputId, input, _ := NewBuffer1D[float32](width)
		addId()
		outputId, output, _ := NewBuffer1D[float32](width)
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
				functionId.Run(Grid{X: 1}, inputId, outputId)
			}
		})
	}
}
