// go:build darwin
//  +build darwin

#include "cache.h"
#include "error.h"
#import <Metal/Metal.h>

extern id<MTLDevice> device;

// Allocate a block of memory accessible to both the CPU and GPU that is large
// enough to hold the number of bytes specified. The buffer is cached and can be
// retrieved with the buffer Id that's returned. A buffer can be supplied as an
// argument to the metal function when the function is run. If any error is
// encountered creating the buffer, this returns 0 and sets an error message in
// error.
int buffer_new(int size, const char **error) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:(size) options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    logError(
        error,
        [NSString
            stringWithFormat:@"Failed to create buffer with %d bytes", size]);
    return 0;
  }

  // Add the buffer to the buffer cache and return its unique Id.
  int bufferId = cache_cache(buffer, error);
  if (bufferId == 0) {
    logError(error, @"Failed to cache buffer");
    return 0;
  }

  return bufferId;
}

// Retrieve a buffer from the cache. If any error is encountered retrieving
// the buffer, this returns nil and sets an error message in error.
void *buffer_retrieve(int bufferId, const char **error) {
  id<MTLBuffer> buffer = cache_retrieve(bufferId, error);
  if (buffer == nil) {
    logError(error, @"Failed to retrieve buffer");
    return nil;
  }

  return [buffer contents];
}

// Free a cached buffer. If any error is encountered relinquishing the memory,
// this sets an error message in error.
void buffer_close(int bufferId, const char **error) {
  id<MTLBuffer> buffer = cache_retrieve(bufferId, error);
  if (buffer == nil) {
    logError(error, @"Failed to retrieve buffer");
    return;
  }

  [buffer setPurgeableState:MTLPurgeableStateEmpty];
  [buffer release];

  cache_remove(bufferId, error);
}