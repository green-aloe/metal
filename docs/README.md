`metal` is a library for running computational tasks (GPGPU) on [Apple silicon](https://en.wikipedia.org/wiki/Apple_silicon) through Apple's [Metal API](https://developer.apple.com/metal/).

# Overview

## Metal (Apple)
Apple's Metal API
is a unified framework
for performing various types of task
on Apple silicon GPUs.
It offers low-level, direct, detailed access
to the hardware (hence, _metal_ )
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
see Apple's documentation [on threads and threadgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931).)

## `metal` (green-aloe)
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

# Documentation

For the full documentation and example usage, see https://pkg.go.dev/github.com%2Fgreen-aloe%2Fmetal?GOOS=darwin.

# TODO