/*
Package metal
is a library
for running computational tasks (GPGPU)
on [Apple silicon]
through Apple's [Metal API].

# Overview

## Metal (Apple)

Apple's Metal API
is a unified framework
for performing various types of task
on Apple silicon GPUs.
It offers low-level, direct, detailed access
to the hardware (hence, "metal" )
for fast and efficient processing.

The processing centers around pipelines,
which consist of
a function to run
and an arbitrary number of arguments and buffers.
The metal function is parsed
into a series of operations,
and the arguments and buffers of data
are streamed through it
in SIMD groups.
(For more details on SIMD groups
and best practices
for writing metal functions using them,
see Apple's documentation [on threads and threadgroups].)

## metal (green-aloe)

This library
leverages Apple's Metal API
to run computational processes
in a distributed, parallel method.
First,
a metal function is parsed,
added to a pipeline,
and cached.
This happens once
for every metal function.
Then,
any number of metal buffers
are created.
A metal buffer
is an array
of arbitrary length
that references items
of an arbitrary type.
The actual type
is defined in the metal function's definition.
Finally,
the metal function
is run
with the metal buffers
and any static arguments.
This streams
the arguments and the data in the buffers
through the computational operation(s)
as sequenced in the metal function.

# Types

This is the mapping
of Go types
to Metal types:

	| Go      | Metal  |
	| ------- | ------ |
	| float32 | float  |
	| N/A     | half   |
	| int32   | int    |
	| int16   | short  |
	| uint32  | uint   |
	| uint16  | ushort |

# Limitations

  - This library
    technically supports only Apple GPUs
    that allow non-uniform threadgroup sizes.
    A table of GPUs and their feature sets can be found on [page 4 here].
    Most support this feature.
    There has been no testing done on GPUs that don't support it.
  - This library
    currently
    does not support non-standard buffers,
    such as buffers
    that are only accessible to the GPU.
    All buffers
    are currently created
    with the same access and performance settings.
  - This library
    is intended specifically
    for running computations (as opposed to renderings).
    This means ths metal functions
    must be kernel functions,
    i.e. prefixed with `kernel`
    and returning `void`.

# Resources

These are some helpful resources
for understanding this process better
and how to use the metal API efficiently:
  - https://adrianhesketh.com/2022/03/31/use-m1-gpu-with-go/
  - https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu

Details on the language specifications
for writing metal functions
can be found in the [MSL Specification].

[Apple silicon]: https://en.wikipedia.org/wiki/Apple_silicon
[Metal API]: https://developer.apple.com/metal/
[on threads and threadgroups]: https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931
[page 4 here]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[MSL Specification]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
*/
package metal
