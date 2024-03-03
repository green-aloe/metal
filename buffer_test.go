//go:build darwin
// +build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// Test_BufferId_Valid tests that BufferId's Valid method correctly identifies a valid buffer Id.
func Test_BufferId_Valid(t *testing.T) {
	// A valid buffer Id has a positive value. Let's run through a bunch of numbers and test that
	// Valid always report the correct status.
	for i := -100_00; i <= 100_000; i++ {
		bufferId := BufferId(i)

		if i > 0 {
			require.True(t, bufferId.Valid())
		} else {
			require.False(t, bufferId.Valid())
		}
	}
}

// Test_NewBuffer_invalid tests that NewBuffer handles invalid arguments correctly.
func Test_NewBuffer_invalid(t *testing.T) {
	t.Run("no width", func(t *testing.T) {
		bufferId, buffer, err := NewBuffer[int32](0)
		require.NotNil(t, err)
		require.Equal(t, "Invalid width", err.Error())
		require.Equal(t, BufferId(0), bufferId)
		require.Nil(t, buffer)
	})

	t.Run("negative width", func(t *testing.T) {
		bufferId, buffer, err := NewBuffer[float32](-1)
		require.NotNil(t, err)
		require.Equal(t, "Invalid width", err.Error())
		require.Equal(t, BufferId(0), bufferId)
		require.Nil(t, buffer)
	})

	t.Run("too many bytes", func(t *testing.T) {
		bufferId, buffer, err := NewBuffer[float32](math.MaxInt32 + 1)
		require.NotNil(t, err)
		require.Equal(t, "Exceeded maximum number of bytes", err.Error())
		require.Equal(t, BufferId(0), bufferId)
		require.Nil(t, buffer)
	})
}

// Test_NewBuffer tests that NewBuffer creates a new metal buffer with the expected underlying type
// and data shape.
func Test_NewBuffer(t *testing.T) {
	// Test the primitive types that satisfy the BufferType constraint.
	testNewBuffer(t, func(i int) byte { return byte(i) })
	testNewBuffer(t, func(i int) rune { return rune(i) })
	testNewBuffer(t, func(i int) uint8 { return uint8(i) })
	testNewBuffer(t, func(i int) uint16 { return uint16(i) })
	testNewBuffer(t, func(i int) uint32 { return uint32(i) })
	testNewBuffer(t, func(i int) uint64 { return uint64(i) })
	testNewBuffer(t, func(i int) int8 { return int8(-i) })
	testNewBuffer(t, func(i int) int16 { return int16(-i) })
	testNewBuffer(t, func(i int) int32 { return int32(-i) })
	testNewBuffer(t, func(i int) int64 { return int64(-i) })
	testNewBuffer(t, func(i int) float32 { return float32(i) * 1.1 })
	testNewBuffer(t, func(i int) float64 { return float64(i) * 1.1 })

	// Test custom types that satisfy the BufferType constraint.
	type MyByte byte
	testNewBuffer(t, func(i int) MyByte { return MyByte(i) })
	type MyRune rune
	testNewBuffer(t, func(i int) MyRune { return MyRune(i) })
	type MyUint8 uint8
	testNewBuffer(t, func(i int) MyUint8 { return MyUint8(i) })
	type MyUint16 uint16
	testNewBuffer(t, func(i int) MyUint16 { return MyUint16(i) })
	type MyUint32 uint32
	testNewBuffer(t, func(i int) MyUint32 { return MyUint32(i) })
	type MyUint64 uint64
	testNewBuffer(t, func(i int) MyUint64 { return MyUint64(i) })
	type MyInt8 int8
	testNewBuffer(t, func(i int) MyInt8 { return MyInt8(-i) })
	type MyInt16 int16
	testNewBuffer(t, func(i int) MyInt16 { return MyInt16(-i) })
	type MyInt32 int32
	testNewBuffer(t, func(i int) MyInt32 { return MyInt32(-i) })
	type MyInt64 int64
	testNewBuffer(t, func(i int) MyInt64 { return MyInt64(-i) })
	type MyFloat32 float32
	testNewBuffer(t, func(i int) MyFloat32 { return MyFloat32(i) * 1.1 })
	type MyFloat64 float64
	testNewBuffer(t, func(i int) MyFloat64 { return MyFloat64(i) * 1.1 })

	t.Run("max size", func(t *testing.T) {
		bufferId, buffer, err := NewBuffer[byte](math.MaxInt32)
		require.Nil(t, err, "Unable to create metal buffer: %s", err)
		require.True(t, validId(bufferId))
		require.Len(t, buffer, math.MaxInt32)
		require.Equal(t, cap(buffer), math.MaxInt32)
	})
}

// testNewBuffer is a helper to test buffer creation for a variety of types.
func testNewBuffer[T BufferType](t *testing.T, converter func(int) T) {
	var a T

	t.Run(fmt.Sprintf("%T one dimension", a), func(t *testing.T) {
		width := rand.Intn(20) + 1

		bufferId, buffer, err := NewBuffer[T](width)
		require.Nil(t, err, "Unable to create metal buffer: %s", err)
		require.True(t, validId(bufferId))
		require.Len(t, buffer, width)
		require.Equal(t, cap(buffer), width)

		// Test that every item in the buffer has its zero value.
		for i := range buffer {
			require.True(t, reflect.ValueOf(buffer[i]).IsZero())
		}

		// Test that we can write to every item in the buffer.
		require.NotPanics(t, func() {
			for i := range buffer {
				buffer[i] = converter(i)
			}
		})

		// Test that every item retained its value.
		for i := range buffer {
			require.Equal(t, converter(i), buffer[i])
		}
	})

	t.Run(fmt.Sprintf("%T two dimensions", a), func(t *testing.T) {
		width := rand.Intn(20) + 1
		height := rand.Intn(20) + 1

		bufferId, buffer1D, err := NewBuffer[T](width * height)
		require.Nil(t, err, "Unable to create metal buffer: %s", err)
		require.True(t, validId(bufferId))
		require.Len(t, buffer1D, width*height)
		require.Equal(t, width*height, cap(buffer1D))

		buffer2D := Fold(buffer1D, width)
		require.Len(t, buffer2D, width)
		require.Equal(t, width, cap(buffer2D))
		for _, y := range buffer2D {
			require.Equal(t, height, len(y))
			require.Equal(t, height, cap(y))
		}

		// Test that every item in the buffer has its zero value.
		for i := range buffer2D {
			for j := range buffer2D[i] {
				require.True(t, reflect.ValueOf(buffer2D[i][j]).IsZero())
			}
		}

		// Test that we can write to every item in the buffer.
		require.NotPanics(t, func() {
			for i := range buffer2D {
				for j := range buffer2D[i] {
					buffer2D[i][j] = converter(i * j)
				}
			}
		})

		// Test that every item retained its value.
		for i := range buffer2D {
			for j := range buffer2D[i] {
				require.Equal(t, converter(i*j), buffer2D[i][j])
			}
		}
	})

	t.Run(fmt.Sprintf("%T three dimensions", a), func(t *testing.T) {
		width := rand.Intn(20) + 1
		height := rand.Intn(20) + 1
		depth := rand.Intn(20) + 1

		bufferId, buffer1D, err := NewBuffer[T](width * height * depth)
		require.Nil(t, err, "Unable to create metal buffer: %s", err)
		require.True(t, validId(bufferId))
		require.Equal(t, width*height*depth, len(buffer1D))
		require.Equal(t, width*height*depth, cap(buffer1D))

		buffer3D := Fold(Fold(buffer1D, width*height), width)
		require.Len(t, buffer3D, width)
		require.Equal(t, width, cap(buffer3D))
		for _, y := range buffer3D {
			require.Equal(t, height, len(y))
			require.Equal(t, height, cap(y))
			for _, z := range y {
				require.Equal(t, depth, len(z))
				require.Equal(t, depth, cap(z))
			}
		}

		// Test that every item in the buffer has its zero value.
		for i := range buffer3D {
			for j := range buffer3D[i] {
				for k := range buffer3D[i][j] {
					require.True(t, reflect.ValueOf(buffer3D[i][j][k]).IsZero())
				}
			}
		}

		// Test that we can write to every item in the buffer.
		require.NotPanics(t, func() {
			for i := range buffer3D {
				for j := range buffer3D[i] {
					for k := range buffer3D[i][j] {
						buffer3D[i][j][k] = converter(i * j * k)
					}
				}
			}
		})

		// Test that every item retained its value.
		for i := range buffer3D {
			for j := range buffer3D[i] {
				for k := range buffer3D[i][j] {
					require.Equal(t, converter(i*j*k), buffer3D[i][j][k])
				}
			}
		}
	})
}

// Test_NewBuffer_threadSafe tests that NewBuffer can handle multiple parallel invocations and still
// return the correct Id.
func Test_NewBuffer_threadSafe(t *testing.T) {
	// We're going to use a wait group to block each goroutine after it's prepared until they're all
	// ready to fire.
	numIter := 100
	var wg sync.WaitGroup
	wg.Add(numIter)

	dataCh := make(chan BufferId)

	// Prepare one goroutine to create a new buffer for each iteration.
	for i := 0; i < numIter; i++ {
		// Calculate the dimensions for this buffer.
		width := rand.Intn(20) + 1

		// Spin up a new goroutine. This will wait until all goroutines are ready to fire, then
		// create a new metal buffer and send its Id back to the main thread.
		go func() {
			wg.Wait()

			bufferId, _, err := NewBuffer[int32](width)
			require.Nil(t, err, "Unable to create metal buffer: %s", err)

			dataCh <- bufferId
		}()

		// Mark that this goroutine is ready.
		wg.Done()
	}

	// Test that each buffer's Id is unique.
	idMap := make(map[BufferId]struct{})
	for i := 0; i < numIter; i++ {
		bufferId := <-dataCh

		_, ok := idMap[bufferId]
		require.False(t, ok)
		idMap[bufferId] = struct{}{}

		addId()
	}

	// Test that we received every Id in the sequence.
	idList := make([]BufferId, 0, len(idMap))
	for bufferId := range idMap {
		idList = append(idList, bufferId)
	}
	sort.Slice(idList, func(i, j int) bool { return idList[i] < idList[j] })
	require.Len(t, idList, numIter)
	for i := 0; i < numIter; i++ {
		require.Equal(t, nextMetalId-numIter+i, int(idList[i]))
	}
}

// Test_NewBufferWith tests that NewBufferWith creates a new metal buffer with the expected
// underlying data.
func Test_NewBufferWith(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		input := []int32{1, 2, 3, 4, 5}
		want := []int32{1, 2, 3, 4, 5}
		bufferId, buffer, err := NewBufferWith(input)
		require.NoError(t, err)
		require.True(t, validId(bufferId))
		require.Len(t, buffer, len(want))
		require.Equal(t, cap(want), cap(buffer))
		require.Equal(t, want, buffer)
	})

	t.Run("float32", func(t *testing.T) {
		input := []float32{1.1, 2.2, 3.3, 4.4, 5.5}
		want := []float32{1.1, 2.2, 3.3, 4.4, 5.5}
		bufferId, buffer, err := NewBufferWith(input)
		require.NoError(t, err)
		require.True(t, validId(bufferId))
		require.Len(t, buffer, len(want))
		require.Equal(t, cap(want), cap(buffer))
		require.Equal(t, want, buffer)
	})
}
