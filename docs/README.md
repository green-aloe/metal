`metal` is a library for running computational tasks (GPGPU) on [Apple silicon](https://en.wikipedia.org/wiki/Apple_silicon) through Apple's [Metal API](https://developer.apple.com/metal/).

# Overview

Apple's Metal API
is a unified framework
for performing various types of task
on Apple silicon GPUs.
It offers low-level, direct, detailed access
to the hardware (hence, _metal_)
for fast and efficient processing.

The processing centers around pipelines,
which consist of
a function to run
and an arbitrary number of buffers.
The metal function is parsed
into a series of operations,
and the buffers of data
are streamed through it
in SIMD groups.
(For more details on SIMD groups
and best practices
for writing metal functions using them,
see Apple's documentation [on threads and threadgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931).)

This library
leverages the Metal API
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
with the metal buffers.
This streams
the data in the buffers
through the computational operation(s)
as sequenced in the metal function.

For the full documentation, see https://pkg.go.dev/github.com%2Fgreen-aloe%2Fmetal?GOOS=darwin.

# TODO