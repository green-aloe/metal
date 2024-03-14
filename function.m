// go:build darwin
//  +build darwin

// The process in this file largely follows the structure detailed in
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu.

#include "cache.h"
#include "error.h"
#import <Metal/Metal.h>

extern id<MTLDevice> device;

// Structure of various metal resources needed to execute a computational
// process on the GPU. We have to bundle this in a header that cgo doesn't
// import because of a bug in LLVM that leads to a compilation error of "struct
// size calculation error off=8 bytesize=0".
typedef struct {
  id<MTLFunction> function;
  id<MTLComputePipelineState> pipeline;
  id<MTLCommandQueue> commandQueue;
} _function;

// Set up a new pipeline for executing the specified function in the provided
// MTL code on the default GPU. This returns an Id that must be used to run the
// function. This should be called only once for every function. If any error is
// encountered initializing the metal function, this returns 0 and sets an error
// message in error.
int function_new(const char *metalCode, const char *funcName,
                 const char **error) {
  if (strlen(metalCode) == 0) {
    logError(error, @"Missing metal code");
    return 0;
  }
  if (strlen(funcName) == 0) {
    logError(error, @"Missing function name");
    return 0;
  }

  // Set up a new function object to hold the various resources for the
  // pipeline.
  _function *function = malloc(sizeof(_function));
  if (function == nil) {
    logError(error, @"Failed to initialize function");
    return 0;
  }

  // Create a new library of metal code, which will be used to get a
  // reference to the function we want to run on the GPU. Normnally, we
  // would use newDefaultLibrary here to automatically create a library from
  // all the .metal files in this package. However, because cgo doesn't have
  // that functionality, we need to use newLibraryWithSource:options:error
  // instead and supply the code to the new library directly.
  NSError *libraryError = nil;
  id<MTLLibrary> library =
      [device newLibraryWithSource:[NSString stringWithUTF8String:metalCode]
                           options:[MTLCompileOptions new]
                             error:&libraryError];
  if (library == nil) {
    logError(error, [NSString stringWithFormat:@"Failed to create library: %@",
                                               libraryError]);
    return 0;
  }

  // Get a reference to the function in the code that's now in the new library.
  // (Note that this is not executable yet. We need a pipeline in order to run
  // this function.)
  function->function =
      [library newFunctionWithName:[NSString stringWithUTF8String:funcName]];
  if (function->function == nil) {
    logError(error, [NSString stringWithFormat:@"Failed to find function '%s'",
                                               funcName]);
    return 0;
  }

  // Convert the function object we just created into a pipeline so we can
  // run the function. A pipeline contains the actual instructions/steps
  // that the GPU uses to execute the code.
  NSError *pipelineError = nil;
  function->pipeline =
      [device newComputePipelineStateWithFunction:function->function
                                            error:&pipelineError];
  if (function->pipeline == nil) {
    logError(error, [NSString stringWithFormat:@"Failed to create pipeline: %@",
                                               pipelineError]);
    return 0;
  }

  // Set up a command queue. This is what sends the work to the GPU.
  function->commandQueue = [device newCommandQueue];
  if (function->commandQueue == nil) {
    logError(error, @"Failed to set up command queue");
    return 0;
  }

  // Save the function for later use and return an Id referencing it.
  int functionId = cache_cache(function);
  if (functionId == 0) {
    logError(error, @"Failed to cache function");
    return 0;
  }

  return functionId;
}

// Execute the computational process on the GPU. Each buffer is supplied as an
// argument to the metal code in the same order as the buffer Ids here. This is
// not thread-safe. If any error is encountered running the metal function, this
// returns false and sets an error message in error.
_Bool function_run(int functionId, int width, int height, int depth,
                   float *inputs, int numInputs, int *bufferIds,
                   int numBufferIds, const char **error) {
  // Fetch the function from the cache.
  _function *function = cache_retrieve(functionId);
  if (function == nil) {
    logError(error, @"Failed to retrieve function");
    return false;
  }

  // Create a command buffer from the command queue in the pipeline. This will
  // hold the processing commands and move through the queue to the GPU.
  id<MTLCommandBuffer> commandBuffer = [function->commandQueue commandBuffer];
  if (commandBuffer == nil) {
    logError(error, @"Failed to set up command buffer");
    return false;
  }

  // Set up an encoder to write the (compute pass) commands and parameters to
  // the command buffer we just created.
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (encoder == nil) {
    logError(error, @"Failed to set up compute encoder");
    return false;
  }

  // Set the pipeline that the encoder will use.
  [encoder setComputePipelineState:function->pipeline];

  // Set the arguments that will be passed to the function. The indexes for the
  // arguments here need to match their order in the function declaration. We'll
  // start with the arguments that are static values, and then we'll add the
  // buffers. We currently support using only the entire buffer without any
  // offsets, which could be used to, say, use one part of a buffer for one
  // function argument and the other part for a different argument.
  int index = 0;
  for (int i = 0; i < numInputs; i++) {
    // Add the static argument bytes to the encoder at the appropriate index.
    [encoder setBytes:&inputs[i] length:sizeof(float) atIndex:index++];
  }
  for (int i = 0; i < numBufferIds; i++) {
    // Retrieve the buffer for this Id.
    id<MTLBuffer> buffer = cache_retrieve(bufferIds[i]);
    if (buffer == nil) {
      logError(
          error,
          [NSString
              stringWithFormat:@"Failed to retrieve buffer %d/%d using Id %d",
                               i + 1, numBufferIds, bufferIds[i]]);
      return false;
    }

    // Add the buffer to the encoder at the appropriate index.
    [encoder setBuffer:buffer offset:0 atIndex:index++];
  }

  // Specify how many threads we need to perform all the calculations (one
  // thread per calculation).
  MTLSize gridSize = MTLSizeMake(width, height, depth);

  // Figure out how many threads will be grouped together into each threadgroup.
  // There are two variables that are important here:
  //
  //     pipeline.threadExecutionWidth:
  //         Maximum number of threads that the GPU can execute at one time in
  //         parallel (aka thread warp size, aka SIMD group size)
  //     pipeline.maxTotalThreadsPerThreadgroup:
  //         Maximum number of threads that can be bundled together into a
  //         threadgroup
  //
  // We're going to divide the threads conceptually into two dimensions and then
  // place them into a 3-dimensional grid with no height. The first dimension
  // will be the number of threads that can run at one time (the thread warp
  // size). The second dimension will be the maximum number of parallel thread
  // bundles.
  //
  // For more details on threads, grids, and threadgroup sizes, see
  // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes.
  NSUInteger w = function->pipeline.threadExecutionWidth;
  NSUInteger h = function->pipeline.maxTotalThreadsPerThreadgroup / w;
  MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

  // Set the grid into the encoder. (With this method, we don't need to
  // calculate the number of threadgroups for the grid.)
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  // Mark that we're done encoding the buffer and can proceed with executing the
  // function.
  [encoder endEncoding];

  // Commit the command buffer to the command queue so that it gets picked up
  // and run on the GPU, and then wait for the calculations to finish.
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  return true;
}

// Get the name of the metal function with the provided function Id, or nil on
// error.
const char *function_name(int functionId) {
  // Fetch the function from the cache.
  _function *function = cache_retrieve(functionId);
  if (function == nil) {
    logError(nil, @"Failed to retrieve function");
    return nil;
  }

  return [[function->function name] UTF8String];
}