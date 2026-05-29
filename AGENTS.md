# Agent Instructions — github.com/green-aloe/metal

This file provides guidance to Claude Code and other AI agents when working with code in this repository.

## Commands

```bash
# Build
go build -v ./...

# Test (darwin only — requires Apple GPU)
go test -v ./...

# Run a single test
go test -v -run TestFunctionName ./...

# Lint
staticcheck ./...
```

## Architecture

`github.com/green-aloe/metal` is a Go library that lets Go programs run Metal GPU compute kernels on macOS via CGo. It is macOS-only — all Go files have `//go:build darwin`.

### Layer structure

**Objective-C / Metal layer** (`.m` / `.h` files):

| File | Purpose |
|------|---------|
| `Metal.h` | C interface declared for Go/CGo to call |
| `Metal.m` | Initializes `MTLDevice` (the GPU handle) and both caches |
| `Function.m` | Compiles MSL source at runtime, manages a compute pipeline per function, dispatches work via a single shared `MTLCommandQueue` |
| `Buffer.m` / `BufferCache.h` | Allocates shared CPU/GPU memory (`MTLResourceStorageModeShared`), typed buffer cache |
| `Error.h` / `Error.m` | `logError()` — writes an error string through a `const char**` out-param back to Go |

**Go / CGo layer**:

| File | Purpose |
|------|---------|
| `function.go` | `NewFunction`, `Function.Valid/String/Run/Close`, `Grid`, `RunParameters` |
| `buffer.go` | `NewBuffer[T]`, `NewBufferWith[T]`, `BufferId.Valid/Close`, `ErrInvalidBufferId` |
| `helpers.go` | `Fold`, `sizeof`, `metalErrToError`, CGo test helpers |
| `doc.go` | Package-level godoc comment (no code) |

### Data flow

1. `NewFunction(src, name)` — compiles MSL source, builds pipeline state, caches it in `Function.m`'s `NSMutableDictionary`, returns `*Function` with an integer ID.
2. `NewBuffer[T](width)` — allocates shared memory via `buffer_new`, returns `BufferId` and a Go slice backed by that memory. One CGo call returns both the ID and the contents pointer.
3. `function.Run(RunParameters{Grid, Inputs, BufferIds})` — encodes the command buffer, dispatches threads, blocks until complete. Safe for concurrent use.
4. `bufferId.Close()` / `function.Close()` — releases GPU resources, zeroes the ID.

### Key design decisions

- **Integer IDs** — all Metal resources are referenced by integer handles (1-based). `0` always means invalid/unset. Functions and buffers have separate caches with independent ID sequences.
- **Typed separate caches** — `Function.m` and `Buffer.m` each own an `NSMutableDictionary` + `NSLock`. No shared untyped cache; a buffer ID cannot accidentally retrieve a function.
- **Single command queue** — one `MTLCommandQueue` is created at init and shared across all `function_run` calls. Each call creates its own `MTLCommandBuffer` (local, not shared), so concurrent `Run` calls are safe.
- **ARC is required** — cgo compiles `.m` files under manual reference counting (MRC) by default. This package depends on ARC to manage Metal object lifetimes, so the Go cgo preambles set `#cgo CFLAGS: -fobjc-arc`, and `Metal.m` has a `#error` guard on `__has_feature(objc_arc)` that fails the build if the flag is ever lost. Do not remove either.
- **`@autoreleasepool` at every cgo entry point** — a Go goroutine calling in through cgo has no ambient autorelease pool, so autoreleased ObjC temporaries (NSStrings, boxed NSNumber keys, command buffers/encoders) would accumulate until the goroutine exits. Each exported C function (`metal_init`, `function_new`/`run`/`name`/`close`, `buffer_new`/`close`) wraps its body in `@autoreleasepool`. The internal cache helpers don't, since they always run inside an entry point's pool.
- **`MetalFunction` ObjC class** — the function struct is an `@interface MetalFunction : NSObject` with `mtlFunction` and `pipeline` properties. ARC manages ObjC object lifetimes automatically, including on error paths.
- **Dispatch path depends on GPU family** — `function_run` uses `dispatchThreads:threadsPerThreadgroup:` (exact, no over-dispatch) only when the device reports non-uniform threadgroup support (`MTLGPUFamilyApple4`/`MTLGPUFamilyMac2`), detected once in `function_cache_init`. On other hardware it falls back to `dispatchThreadgroups:threadsPerThreadgroup:` with a rounded-up threadgroup count, and kernels must bounds-check their thread position. The threadgroup height is clamped to ≥1 to avoid an invalid zero-dimension threadgroup for high-register-pressure pipelines.
- **No `convertList` copy** — `Run` passes `Inputs` and `BufferIds` to C via direct `unsafe.Pointer` cast. `float32`/`C.float` and `int32`/`C.int` are binary compatible on all Apple platforms.
- **`Fold` semantics** — partitions by column, not row. `Fold(buf, width)` → `width` sub-slices each of length `N/width`. So `buf2D[x][y]` = index `x*(N/width)+y`. This matches how Metal grid coordinates map to buffer indices.
- **`Inputs []float32`** — static scalar args passed before buffer args in the shader. Go always sends `float32` bits; the Metal shader's type declaration governs how they're interpreted (`constant int *`, `constant float *`, etc.).
- **Go type → Metal type**: `float32`↔`float`, `int32`↔`int`, `int16`↔`short`, `uint32`↔`uint`, `uint16`↔`ushort`.

### Error handling

- ObjC functions report errors via `const char **error` out-params, filled by `logError()` in `Error.m`.
- Go side converts via `metalErrToError(err, wrap)` in `helpers.go` — wraps the ObjC message in a Go error.
- Sentinel errors: `ErrInvalidBufferId` and `ErrInvalidFunctionId` — callers can use `errors.Is`.

### Testing

- Tests are in `function_test.go`, `buffer_test.go`, `helpers_test.go`, `example_test.go`.
- `function_test.go` owns the global `nextFunctionId` / `nextBufferId` counters and `validFunctionId` / `validBufferId` helpers — these verify the caches assign sequential IDs correctly. All tests that allocate Metal resources must update these counters.
- The ObjC layer has no separate test framework — it is fully exercised through the Go tests via CGo.
- 95.9% statement coverage. The uncovered branches all require an environment the CI/dev GPU does not exhibit and cannot be triggered in a test: the `ErrMetalUnavailable` guards in `NewBuffer`/`NewFunction`/`Run` and the device allocation failure path in `NewBuffer` (Metal is always available on the test machine); the `else` (uniform-grid) dispatch fallback in `function_run`, which only runs on GPUs without non-uniform threadgroup support; and the `h == 0` threadgroup-height clamp, which needs a kernel whose `maxTotalThreadsPerThreadgroup` is below the device's `threadExecutionWidth`.
- `staticcheck.conf` suppresses `-ST1005` (error string casing) and `-ST1003` (naming — accommodates CGo identifiers).

### Gotchas

- `Fold` return value shares the backing array with the input slice — mutating the original slice mutates the folded view and vice versa. This is intentional.
- After `Close()`, the ID is zeroed. Calling `Run` or `Close` on a closed function/buffer returns an error; it does not panic.
- `NewFunction` and `NewBuffer` are safe for concurrent use. `Run` is safe for concurrent use. `Close` is NOT safe to call concurrently with `Run` on the same resource — the caller must coordinate.
- The `//go:build darwin` constraint means this package produces no code on non-macOS platforms. Importing it in a cross-platform binary is fine as long as usages are also guarded.
