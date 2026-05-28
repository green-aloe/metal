`metal` is a Go library for running GPU compute kernels on macOS via Apple's [Metal API](https://developer.apple.com/metal/). It lets you write shader code in [Metal Shading Language (MSL)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), allocate shared CPU/GPU memory, and dispatch parallel computations — all from Go.

## Requirements

- macOS on Apple silicon (M1 or later)
- Go 1.18+ (uses generics)
- Xcode Command Line Tools (provides the Metal compiler and frameworks)

## Installation

```bash
go get github.com/green-aloe/metal
```

## Quick start

```go
import "github.com/green-aloe/metal"

// 1. Write a Metal shader in MSL
source := `
    #include <metal_stdlib>
    using namespace metal;

    kernel void square(device float *data, uint pos [[thread_position_in_grid]]) {
        data[pos] = data[pos] * data[pos];
    }
`

// 2. Compile the shader once
fn, err := metal.NewFunction(source, "square")
if err != nil {
    log.Fatal(err)
}
defer fn.Close()

// 3. Allocate shared CPU/GPU memory
n := 1000
id, buf, err := metal.NewBuffer[float32](n)
if err != nil {
    log.Fatal(err)
}
defer id.Close()

// 4. Write inputs directly into the slice
for i := range buf {
    buf[i] = float32(i)
}

// 5. Run on the GPU — buf is updated in place
err = fn.Run(metal.RunParameters{
    Grid:      metal.Grid{X: n},
    BufferIds: []metal.BufferId{id},
})
if err != nil {
    log.Fatal(err)
}

// 6. Read results directly from the slice
fmt.Println(buf[:5]) // [0 1 4 9 16]
```

## How it works

Apple's unified memory architecture means the CPU and GPU share the same RAM. `NewBuffer` allocates a block of that memory and gives you back both a Go slice (for CPU access) and a `BufferId` (to pass to the shader). There's no copy — the GPU reads and writes the same memory the Go slice points to.

`NewFunction` compiles your MSL source to a GPU pipeline at runtime. Call it once per shader and reuse the returned `*Function` as many times as needed.

`Run` blocks until the GPU finishes. It's safe for concurrent use — multiple goroutines can call `Run` on the same function simultaneously.

## Buffers and dimensions

Buffers are always allocated as a flat 1D slice. Use `Fold` to create a 2D or 3D view over the same memory without copying:

```go
// 2D: 100 columns × 50 rows
_, flat, _ := metal.NewBuffer[float32](100 * 50)
grid2D := metal.Fold(flat, 100)  // grid2D[x][y]

// 3D: 10 × 20 × 30
_, flat, _ = metal.NewBuffer[float32](10 * 20 * 30)
grid3D := metal.Fold(metal.Fold(flat, 10*20), 10)  // grid3D[x][y][z]
```

`Fold(buf, width)` partitions by column: it produces `width` sub-slices each of length `N/width`, so `grid2D[x][y]` maps to flat index `x*(N/width)+y`.

## Static scalar inputs

`RunParameters.Inputs` lets you pass constant scalar values to the shader without allocating a buffer. They're passed as the first arguments to the kernel, before the buffers:

```go
fn.Run(metal.RunParameters{
    Grid:      metal.Grid{X: n},
    Inputs:    []float32{2.0},  // passed as first arg
    BufferIds: []metal.BufferId{inputId, outputId},
})
```

Go always sends inputs as `float32` bits. The Metal shader's parameter type governs how they're interpreted — `constant float *`, `constant int *`, etc.

## Type mapping

| Go type | Metal type |
|---------|-----------|
| `float32` | `float` |
| `int32` | `int` |
| `int16` | `short` |
| `uint32` | `uint` |
| `uint16` | `ushort` |

There is no Go equivalent for Metal's `half` (16-bit float).

## Concurrency

| Operation | Concurrent safe? |
|-----------|-----------------|
| `NewFunction` | Yes |
| `NewBuffer` / `NewBufferWith` | Yes |
| `Function.Run` | Yes — multiple goroutines can call Run on the same Function |
| `BufferId.Close` / `Function.Close` | No — do not call Close while Run is in progress on the same resource |

## Limitations

- macOS on Apple silicon only — the library does not compile on other platforms.
- All buffers use shared CPU/GPU memory (`MTLResourceStorageModeShared`). GPU-private buffers are not supported.
- Only compute kernels are supported (`kernel void` functions). Vertex and fragment shaders are not.
- Requires Apple GPUs that support non-uniform threadgroup sizes (all M-series chips do). See [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) page 4.
- MSL source is compiled at runtime — there is no support for pre-compiled `.metallib` files.

## Full documentation

[pkg.go.dev/github.com/green-aloe/metal](https://pkg.go.dev/github.com%2Fgreen-aloe%2Fmetal?GOOS=darwin)
