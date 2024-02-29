//go:build darwin
// +build darwin

package metal

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// Test_sizeof tests that sizeof returns the correct size for various types.
func Test_sizeof(t *testing.T) {
	// Boolean
	var boolean bool
	require.Equal(t, int(unsafe.Sizeof(boolean)), sizeof[bool]())
	require.Equal(t, int(reflect.TypeOf(boolean).Size()), sizeof[bool]())
	require.Equal(t, 1, sizeof[bool]())

	// Bytes and strings
	var b byte
	require.Equal(t, int(unsafe.Sizeof(b)), sizeof[byte]())
	require.Equal(t, int(reflect.TypeOf(b).Size()), sizeof[byte]())
	require.Equal(t, 1, sizeof[byte]()) // alias for uint8
	var r rune
	require.Equal(t, int(unsafe.Sizeof(r)), sizeof[rune]())
	require.Equal(t, int(reflect.TypeOf(r).Size()), sizeof[rune]())
	require.Equal(t, 4, sizeof[rune]()) // alias for int32
	var s string
	require.Equal(t, int(unsafe.Sizeof(s)), sizeof[string]())
	require.Equal(t, int(reflect.TypeOf(s).Size()), sizeof[string]())
	require.Equal(t, 16, sizeof[string]())
	s = "words"
	require.Equal(t, int(unsafe.Sizeof(s)), sizeof[string]())
	require.Equal(t, int(reflect.TypeOf(s).Size()), sizeof[string]())

	// Unsigned integers
	var u8 uint8
	require.Equal(t, int(unsafe.Sizeof(u8)), sizeof[uint8]())
	require.Equal(t, int(reflect.TypeOf(u8).Size()), sizeof[uint8]())
	require.Equal(t, 1, sizeof[uint8]())
	var u16 uint16
	require.Equal(t, int(unsafe.Sizeof(u16)), sizeof[uint16]())
	require.Equal(t, int(reflect.TypeOf(u16).Size()), sizeof[uint16]())
	require.Equal(t, 2, sizeof[uint16]())
	var u32 uint32
	require.Equal(t, int(unsafe.Sizeof(u32)), sizeof[uint32]())
	require.Equal(t, int(reflect.TypeOf(u32).Size()), sizeof[uint32]())
	require.Equal(t, 4, sizeof[uint32]())
	var u64 uint64
	require.Equal(t, int(unsafe.Sizeof(u64)), sizeof[uint64]())
	require.Equal(t, int(reflect.TypeOf(u64).Size()), sizeof[uint64]())
	require.Equal(t, 8, sizeof[uint64]())

	// Signed integers
	var i8 int8
	require.Equal(t, int(unsafe.Sizeof(i8)), sizeof[int8]())
	require.Equal(t, int(reflect.TypeOf(i8).Size()), sizeof[int8]())
	require.Equal(t, 1, sizeof[int8]())
	var i16 int16
	require.Equal(t, int(unsafe.Sizeof(i16)), sizeof[int16]())
	require.Equal(t, int(reflect.TypeOf(i16).Size()), sizeof[int16]())
	require.Equal(t, 2, sizeof[int16]())
	var i32 int32
	require.Equal(t, int(unsafe.Sizeof(i32)), sizeof[int32]())
	require.Equal(t, int(reflect.TypeOf(i32).Size()), sizeof[int32]())
	require.Equal(t, 4, sizeof[int32]())
	var i64 int64
	require.Equal(t, int(unsafe.Sizeof(i64)), sizeof[int64]())
	require.Equal(t, int(reflect.TypeOf(i64).Size()), sizeof[int64]())
	require.Equal(t, 8, sizeof[int64]())

	// Other integers
	var u uint
	require.Equal(t, int(unsafe.Sizeof(u)), sizeof[uint]())
	require.Equal(t, int(reflect.TypeOf(u).Size()), sizeof[uint]())
	require.Equal(t, 8, sizeof[uint]()) // either 32 or 64 bits, but most machines use 64-bit architecture these days
	var i int
	require.Equal(t, int(unsafe.Sizeof(i)), sizeof[int]())
	require.Equal(t, int(reflect.TypeOf(i).Size()), sizeof[int]())
	require.Equal(t, 8, sizeof[int]()) // same as uint
	var ptr uintptr
	require.Equal(t, int(unsafe.Sizeof(ptr)), sizeof[uintptr]())
	require.Equal(t, int(reflect.TypeOf(ptr).Size()), sizeof[uintptr]())
	require.Equal(t, 8, sizeof[uintptr]())
	var pint *int
	require.Equal(t, int(unsafe.Sizeof(pint)), sizeof[*int]())
	require.Equal(t, int(reflect.TypeOf(pint).Size()), sizeof[*int]())
	require.Equal(t, 8, sizeof[*int]()) // same as uintptr
	pint = new(int)
	*pint = 10
	require.Equal(t, int(unsafe.Sizeof(pint)), sizeof[*int]())
	require.Equal(t, int(reflect.TypeOf(pint).Size()), sizeof[*int]())

	// Floating-point numbers
	var flt32 float32
	require.Equal(t, int(unsafe.Sizeof(flt32)), sizeof[float32]())
	require.Equal(t, int(reflect.TypeOf(flt32).Size()), sizeof[float32]())
	require.Equal(t, 4, sizeof[float32]())
	var flt64 float64
	require.Equal(t, int(unsafe.Sizeof(flt64)), sizeof[float64]())
	require.Equal(t, int(reflect.TypeOf(flt64).Size()), sizeof[float64]())
	require.Equal(t, 8, sizeof[float64]())

	// Complex numbers
	var cmplx64 complex64
	require.Equal(t, int(unsafe.Sizeof(cmplx64)), sizeof[complex64]())
	require.Equal(t, int(reflect.TypeOf(cmplx64).Size()), sizeof[complex64]())
	require.Equal(t, 8, sizeof[complex64]())
	var cmplx128 complex128
	require.Equal(t, int(unsafe.Sizeof(cmplx128)), sizeof[complex128]())
	require.Equal(t, int(reflect.TypeOf(cmplx128).Size()), sizeof[complex128]())
	require.Equal(t, 16, sizeof[complex128]())

	// Arrays and slices
	var a10 [10]int16
	require.Equal(t, int(unsafe.Sizeof(a10)), sizeof[[10]int16]())
	require.Equal(t, int(reflect.TypeOf(a10).Size()), sizeof[[10]int16]())
	require.Equal(t, 20, sizeof[[10]int16]()) // 2 bytes * 10
	var slc []uint64
	require.Equal(t, int(unsafe.Sizeof(slc)), sizeof[[]uint64]())
	require.Equal(t, int(reflect.TypeOf(slc).Size()), sizeof[[]uint64]())
	require.Equal(t, 24, sizeof[[]uint64]()) // slice header = uintptr + (2 * int), or 8 + (2 * 8)
	slc = []uint64{0, 1, 2, 3, 4}
	require.Equal(t, int(unsafe.Sizeof(slc)), sizeof[[]uint64]())
	require.Equal(t, int(reflect.TypeOf(slc).Size()), sizeof[[]uint64]())
	require.Equal(t, 24, sizeof[[]uint64]()) // slice header = uintptr + (2 * int), or 8 + (2 * 8)

	// Struct
	type MyStruct1 struct {
		_ int32
	}
	var strct1 MyStruct1
	require.Equal(t, int(unsafe.Sizeof(strct1)), sizeof[MyStruct1]())
	require.Equal(t, int(reflect.TypeOf(strct1).Size()), sizeof[MyStruct1]())
	require.Equal(t, 4, sizeof[MyStruct1]()) // same as int32
	type MyStruct2 struct {
		_ byte
		_ float32
		_ []int
	}
	var strct2 MyStruct2
	require.Equal(t, int(unsafe.Sizeof(strct2)), sizeof[MyStruct2]())
	require.Equal(t, int(reflect.TypeOf(strct2).Size()), sizeof[MyStruct2]())
	require.Equal(t, 32, sizeof[MyStruct2]()) // same as byte + float32 + slice header + padding

	// Type alias
	type float32Alias float32
	var flt32Alias float32Alias
	require.Equal(t, int(unsafe.Sizeof(flt32Alias)), sizeof[float32Alias]())
	require.Equal(t, int(reflect.TypeOf(flt32Alias).Size()), sizeof[float32Alias]())
	require.Equal(t, 4, sizeof[float32Alias]())

	// Interfaces
	type MyInterface interface{ Method(int) string }
	var iface MyInterface
	require.Equal(t, int(unsafe.Sizeof(iface)), sizeof[MyInterface]())
	require.Equal(t, 16, sizeof[MyInterface]()) // same as 2 * uintptr (one pointer for data, one pointer for methods table)
	var a any
	require.Equal(t, int(unsafe.Sizeof(a)), sizeof[any]())
	require.Equal(t, 16, sizeof[any]())
	var err error
	require.Equal(t, int(unsafe.Sizeof(err)), sizeof[error]())
	require.Equal(t, 16, sizeof[error]())

	// Map
	var m map[string]string
	require.Equal(t, int(unsafe.Sizeof(m)), sizeof[map[string]string]())
	require.Equal(t, int(reflect.TypeOf(m).Size()), sizeof[map[string]string]())
	require.Equal(t, 8, sizeof[map[string]string]()) // same as uintptr
	m = make(map[string]string)
	require.Equal(t, int(unsafe.Sizeof(m)), sizeof[func()]())
	require.Equal(t, int(reflect.TypeOf(m).Size()), sizeof[func()]())
	_ = m

	// Channel
	var ch chan int8
	require.Equal(t, int(unsafe.Sizeof(ch)), sizeof[chan int8]())
	require.Equal(t, int(reflect.TypeOf(ch).Size()), sizeof[chan int8]())
	require.Equal(t, 8, sizeof[chan int8]()) // same as uintptr
	ch = make(chan int8)
	require.Equal(t, int(unsafe.Sizeof(ch)), sizeof[chan int8]())
	require.Equal(t, int(reflect.TypeOf(ch).Size()), sizeof[chan int8]())
	_ = ch

	// Function
	var fn func()
	require.Equal(t, int(unsafe.Sizeof(fn)), sizeof[func()]())
	require.Equal(t, int(reflect.TypeOf(fn).Size()), sizeof[func()]())
	require.Equal(t, 8, sizeof[func()]()) // same as uintptr
	fn = func() { fmt.Println("Hello, world") }
	require.Equal(t, int(unsafe.Sizeof(fn)), sizeof[func()]())
	require.Equal(t, int(reflect.TypeOf(fn).Size()), sizeof[func()]())
	_ = fn
}

// Test_fold tests that fold correctly portions up slices of varying widths.
func Test_fold(t *testing.T) {
	list := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Test that fold returns nil when given a nil or empty list.
	require.Nil(t, fold[int](nil, 10))
	require.Nil(t, fold([]int{}, 10))

	// Test that fold returns nil when the width is not a positive number.
	require.Nil(t, fold(list, -1))
	require.Nil(t, fold(list, 0))

	// Test that fold returns nil when the width does not evenly divide the list.
	require.Nil(t, fold(list, 3))

	// Test running fold on a number of lists of different types, and also test that the output is
	// just a wrapper around the input and not new memory.

	// Test folding 4 integers into 2 groups of 2.
	input1 := []int{1, 2, 3, 4}
	want1 := [][]int{{1, 2}, {3, 4}}
	have1 := fold(input1, 2)
	require.Equal(t, want1, have1)
	for i := range have1 {
		require.Equal(t, 2, cap(have1[i]))
	}

	// Test that the folded slice still references the original backing array.
	for i := range input1 {
		input1[i] = i + 11
	}
	want1 = [][]int{{11, 12}, {13, 14}}
	require.Equal(t, want1, have1)
	for i := range have1 {
		require.Equal(t, 2, cap(have1[i]))
	}

	// Test folding 8 floats into 2 groups of 4.
	input2 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	want2 := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	have2 := fold(input2, 2)
	require.Equal(t, want2, have2)
	for i := range have2 {
		require.Equal(t, 4, cap(have2[i]))
	}

	// Test that the folded slice still references the original backing array.
	for i := range input2 {
		input2[i] = float32(i + 11)
	}
	want2 = [][]float32{{11, 12, 13, 14}, {15, 16, 17, 18}}
	require.Equal(t, want2, have2)
	for i := range have2 {
		require.Equal(t, 4, cap(have2[i]))
	}

	// Test folding 7 integers into 1 group of 7.
	input3 := []int8{1, 2, 3, 4, 5, 6, 7}
	want3 := [][]int8{{1, 2, 3, 4, 5, 6, 7}}
	have3 := fold(input3, 1)
	require.Equal(t, want3, have3)
	for i := range have3 {
		require.Equal(t, 7, cap(have3[i]))
	}

	// Test that the folded slice still references the original backing array.
	for i := range input3 {
		input3[i] = int8(i + 11)
	}
	want3 = [][]int8{{11, 12, 13, 14, 15, 16, 17}}
	require.Equal(t, want3, have3)
	for i := range have3 {
		require.Equal(t, 7, cap(have3[i]))
	}

	// Test folding 7 integers into 7 groups of 1.
	input4 := []int8{1, 2, 3, 4, 5, 6, 7}
	want4 := [][]int8{{1}, {2}, {3}, {4}, {5}, {6}, {7}}
	have4 := fold(input4, 7)
	require.Equal(t, want4, have4)
	for i := range have4 {
		require.Equal(t, 1, cap(have4[i]))
	}

	// Test that the folded slice still references the original backing array.
	for i := range input4 {
		input4[i] = int8(i + 11)
	}
	want4 = [][]int8{{11}, {12}, {13}, {14}, {15}, {16}, {17}}
	require.Equal(t, want4, have4)
	for i := range have4 {
		require.Equal(t, 1, cap(have4[i]))
	}

	// Test folding 24 integers into 8 groups of 3.
	input5 := []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
	want5 := [][]uint64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}
	have5 := fold(input5, 8)
	require.Equal(t, want5, have5)
	for i := range have5 {
		require.Equal(t, 3, cap(have5[i]))
	}

	// Test folding those 8 slices from above into 2 groups of 4.
	want6 := [][][]uint64{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}, {{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}}
	have6 := fold(have5, 2)
	require.Equal(t, want6, have6)
	for i := range have6 {
		require.Equal(t, 4, cap(have6[i]))
		for j := range have6[i] {
			require.Equal(t, 3, cap(have6[i][j]))
		}
	}

	// Test that the folded slices still reference the original backing array.
	for i := range input5 {
		input5[i] = uint64(i + 11)
	}
	want5 = [][]uint64{{11, 12, 13}, {14, 15, 16}, {17, 18, 19}, {20, 21, 22}, {23, 24, 25}, {26, 27, 28}, {29, 30, 31}, {32, 33, 34}}
	require.Equal(t, want5, have5)
	want6 = [][][]uint64{{{11, 12, 13}, {14, 15, 16}, {17, 18, 19}, {20, 21, 22}}, {{23, 24, 25}, {26, 27, 28}, {29, 30, 31}, {32, 33, 34}}}
	require.Equal(t, want6, have6)
}

// Test_convertList tests that convertList correctly converts lists from one type to another and
// returns a pointer to the first element (if any).
func Test_convertList(t *testing.T) {
	t.Run("nil list", func(t *testing.T) {
		outputs, outputsPtr := convertList[int32, int32](nil)
		require.Nil(t, outputs)
		require.Nil(t, outputsPtr)
	})

	t.Run("empty list", func(t *testing.T) {
		outputs, outputsPtr := convertList[int32, int32]([]int32{})
		require.Nil(t, outputs)
		require.Nil(t, outputsPtr)
	})

	t.Run("list with one element", func(t *testing.T) {
		inputs := []int32{1}
		outputs, outputsPtr := convertList[int32, int32](inputs)
		require.Equal(t, inputs, outputs)
		require.Equal(t, &inputs[0], outputsPtr)
	})

	t.Run("list with multiple elements", func(t *testing.T) {
		inputs := []int32{1, 2, 3}
		outputs, outputsPtr := convertList[int32, int32](inputs)
		require.Equal(t, inputs, outputs)
		require.Equal(t, &inputs[0], outputsPtr)
	})

	t.Run("conversion from int32 to float32", func(t *testing.T) {
		inputs := []int32{1, 2, 3}
		want := []float32{1.0, 2.0, 3.0}
		outputs, outputsPtr := convertList[int32, float32](inputs)
		require.Equal(t, want, outputs)
		require.Equal(t, &want[0], outputsPtr)
	})

	t.Run("conversion from int32 to float64", func(t *testing.T) {
		inputs := []int32{1, 2, 3}
		want := []float64{1.0, 2.0, 3.0}
		outputs, outputsPtr := convertList[int32, float64](inputs)
		require.Equal(t, want, outputs)
		require.Equal(t, &want[0], outputsPtr)
	})

	t.Run("conversion from float32 to int32", func(t *testing.T) {
		inputs := []float32{1.0, 2.0, 3.0}
		want := []int32{1, 2, 3}
		outputs, outputsPtr := convertList[float32, int32](inputs)
		require.Equal(t, want, outputs)
		require.Equal(t, &want[0], outputsPtr)
	})

	t.Run("conversion from float64 to int32", func(t *testing.T) {
		inputs := []float64{1.0, 2.0, 3.0}
		want := []int32{1, 2, 3}
		outputs, outputsPtr := convertList[float64, int32](inputs)
		require.Equal(t, want, outputs)
		require.Equal(t, &want[0], outputsPtr)
	})
}

// Test_metalErrToError tests that metalErrToError returns a go error that wraps a metal error.
func Test_metalErrToError(t *testing.T) {
	type subtest struct {
		metalErr string
		goErr    string
		want     string
	}

	subtests := []subtest{
		{
			// Nothing
		},
		{
			// Only wrapping
			goErr: "go error",
			want:  "go error",
		},
		{
			// Metal error with no wrapping
			metalErr: "metal error",
			want:     "metal error",
		},
		{
			// Metal error wrapped in a go error
			metalErr: "metal error",
			goErr:    "go error",
			want:     "go error: metal error",
		},
	}

	for i, subtest := range subtests {
		t.Run(fmt.Sprintf("Subtest%d_metalErr='%s'_goErr='%s'", i+1, subtest.metalErr, subtest.goErr), func(t *testing.T) {
			// Create a C string to mimic how an error would be returned from the metal functions.
			metalErr := cgoString(subtest.metalErr)
			defer cgoFree(metalErr)

			// Run any errors we have for this subtest through the helper.
			err := metalErrToError(metalErr, subtest.goErr)

			// If we don't have any error messages, then the error should be nil. Otherwise, we should
			// have received the expected formatted error.
			if subtest.metalErr == "" && subtest.goErr == "" {
				require.Nil(t, err)
			} else {
				require.NotNil(t, err)
				require.Equal(t, subtest.want, err.Error())

				// If we have both error messages, then we should be able to unwrap the error to get the
				// underlying metal error. Otherwise, the error shouldn't be wrapped at all.
				unwrapErr := errors.Unwrap(err)
				if subtest.metalErr != "" && subtest.goErr != "" {
					require.NotNil(t, unwrapErr)
					require.Equal(t, subtest.metalErr, unwrapErr.Error())
				} else {
					require.Nil(t, unwrapErr)
				}
			}
		})
	}
}
