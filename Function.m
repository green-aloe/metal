// The process in this file largely follows the structure detailed in
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu.

#include "BufferCache.h"
#include "Error.h"
#include "FunctionCache.h"
#include "Metal.h"
#include "MetalInternal.h"
#include <limits.h>
#include <string.h>
#import <Metal/Metal.h>

// ObjC class holding Metal resources for a compute pipeline. Storing this as
// an ObjC object (vs. a malloc'd C struct boxed in NSValue) lets ARC manage
// the lifetime of the MTLFunction and MTLComputePipelineState members
// automatically, including on error paths.
@interface MetalFunction : NSObject
@property (nonatomic, strong) id<MTLFunction> mtlFunction;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@end

@implementation MetalFunction
@end

static NSMutableDictionary *functionCache = nil;
static int nextFunctionId = 1;
static NSLock *functionLock = nil;
static id<MTLCommandQueue> commandQueue = nil;

// Initialize the function cache and shared command queue. This should be called
// only once. Returns false if the command queue could not be created.
_Bool function_cache_init(void) {
  functionCache = [[NSMutableDictionary alloc] init];
  functionLock = [[NSLock alloc] init];
  commandQueue = [metal_device() newCommandQueue];
  return commandQueue != nil;
}

// Set up a new pipeline for executing the specified function in the provided
// MTL code on the default GPU. This returns an Id that must be used to run the
// function. This should be called only once for every function. If any error is
// encountered initializing the metal function, this returns 0 and sets an error
// message in error.
int function_new(const char *metalCode, const char *funcName,
                 const char **error) {
  if (strlen(metalCode) == 0) {
    logError(error, @"missing metal code");
    return 0;
  }
  if (strlen(funcName) == 0) {
    logError(error, @"missing function name");
    return 0;
  }

  // Set up a new function object to hold the various resources for the
  // pipeline.
  MetalFunction *function = [[MetalFunction alloc] init];
  if (function == nil) {
    logError(error, @"failed to initialize function");
    return 0;
  }

  // Create a new library of metal code, which will be used to get a
  // reference to the function we want to run on the GPU. Normally, we
  // would use newDefaultLibrary here to automatically create a library from
  // all the .metal files in this package. However, because cgo doesn't have
  // that functionality, we need to use newLibraryWithSource:options:error
  // instead and supply the code to the new library directly.
  NSError *libraryError = nil;
  id<MTLLibrary> library =
      [metal_device() newLibraryWithSource:[NSString stringWithUTF8String:metalCode]
                                   options:nil
                                     error:&libraryError];
  if (library == nil) {
    logError(error, [NSString stringWithFormat:@"failed to create library: %@",
                                               libraryError]);
    function = nil;
    return 0;
  }

  // Get a reference to the function in the code that's now in the new library.
  // (Note that this is not executable yet. We need a pipeline in order to run
  // this function.)
  function.mtlFunction =
      [library newFunctionWithName:[NSString stringWithUTF8String:funcName]];
  if (function.mtlFunction == nil) {
    logError(error, [NSString stringWithFormat:@"failed to find function '%s'",
                                               funcName]);
    function = nil;
    return 0;
  }

  // Convert the function object we just created into a pipeline so we can
  // run the function. A pipeline contains the actual instructions/steps
  // that the GPU uses to execute the code.
  NSError *pipelineError = nil;
  function.pipeline =
      [metal_device() newComputePipelineStateWithFunction:function.mtlFunction
                                                    error:&pipelineError];
  if (function.pipeline == nil) {
    logError(error, [NSString stringWithFormat:@"failed to create pipeline: %@",
                                               pipelineError]);
    function = nil;
    return 0;
  }

  // Store the function in the cache under the next available ID.
  int functionId = 0;
  [functionLock lock];
  if (nextFunctionId == INT_MAX) {
    [functionLock unlock];
    logError(error, @"function id space exhausted");
    return 0;
  }
  functionId = nextFunctionId++;
  functionCache[@(functionId)] = function;
  [functionLock unlock];

  return functionId;
}

// Execute the computational process on the GPU. Each buffer is supplied as an
// argument to the metal code in the same order as the buffer Ids here. This is
// safe for concurrent use: each call creates its own command buffer and encoder,
// and the command queue serializes GPU execution automatically. If any error is
// encountered running the metal function, this returns false and sets an error
// message in error.
_Bool function_run(int functionId, unsigned int width, unsigned int height,
                   unsigned int depth, float *inputs, int numInputs,
                   int *bufferIds, int numBufferIds, const char **error) {
  // Fetch the function from the cache.
  [functionLock lock];
  MetalFunction *function = functionCache[@(functionId)];
  [functionLock unlock];

  if (function == nil) {
    logError(error, [NSString stringWithFormat:@"failed to retrieve function: invalid cache id: %d", functionId]);
    return false;
  }

  // Create a command buffer from the shared command queue. This will hold the
  // processing commands and move through the queue to the GPU.
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  if (commandBuffer == nil) {
    logError(error, @"failed to set up command buffer");
    return false;
  }

  // Set up an encoder to write the (compute pass) commands and parameters to
  // the command buffer we just created.
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (encoder == nil) {
    logError(error, @"failed to set up compute encoder");
    return false;
  }

  // Set the pipeline that the encoder will use.
  [encoder setComputePipelineState:function.pipeline];

  // Set the arguments that will be passed to the function. The indexes for the
  // arguments here need to match their order in the function declaration. We'll
  // start with the arguments that are static values, and then we'll add the
  // buffers. We currently support using only the entire buffer without any
  // offsets, which could be used to, say, use one part of a buffer for one
  // function argument and the other part for a different argument.
  int index = 0;
  for (int i = 0; i < numInputs; i++) {
    [encoder setBytes:&inputs[i] length:sizeof(float) atIndex:index++];
  }
  for (int i = 0; i < numBufferIds; i++) {
    // Retrieve the buffer for this Id from the buffer cache.
    id<MTLBuffer> buffer = buffer_cache_retrieve(bufferIds[i]);
    if (buffer == nil) {
      // Close out the encoder before bailing — Metal requires endEncoding
      // before the encoder is released, even on the error path.
      [encoder endEncoding];
      logError(error,
               [NSString stringWithFormat:@"failed to retrieve buffer %d/%d: invalid cache id: %d",
                                          i + 1, numBufferIds, bufferIds[i]]);
      return false;
    }

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
  NSUInteger w = function.pipeline.threadExecutionWidth;
  NSUInteger h = function.pipeline.maxTotalThreadsPerThreadgroup / w;
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

// Get the name of the metal function with the provided function Id, or NULL on
// error. The returned C string is heap-allocated (strdup); the caller (Go side)
// must free it. We strdup rather than return -[NSString UTF8String] directly
// because that pointer's lifetime is tied to the current autorelease pool and
// is not safe to hand across the cgo boundary.
const char *function_name(int functionId) {
  [functionLock lock];
  MetalFunction *function = functionCache[@(functionId)];
  [functionLock unlock];

  if (function == nil) {
    return NULL;
  }

  const char *bytes = [[function.mtlFunction name] UTF8String];
  if (bytes == NULL) {
    return NULL;
  }

  return strdup(bytes);
}

// Release the compiled pipeline for the given function Id. After this call the
// Id is invalid. Returns false and sets an error if the Id is not found.
_Bool function_close(int functionId, const char **error) {
  [functionLock lock];
  MetalFunction *function = functionCache[@(functionId)];
  if (function == nil) {
    [functionLock unlock];
    logError(error, [NSString stringWithFormat:@"invalid function id: %d", functionId]);
    return false;
  }
  [functionCache removeObjectForKey:@(functionId)];
  [functionLock unlock];

  return true;
}
