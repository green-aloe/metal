#import "BufferCache.h"
#import "Error.h"
#import "Metal.h"
#import "MetalInternal.h"
#include <limits.h>
#import <Metal/Metal.h>

static NSMutableDictionary *bufferCache = nil;
static int nextBufferId = 1;
static NSLock *bufferLock = nil;

// Initialize the buffer cache. This should be called only once.
void buffer_cache_init(void) {
  bufferCache = [[NSMutableDictionary alloc] init];
  bufferLock = [[NSLock alloc] init];
}

// Store a buffer in the cache and return its ID, or 0 on error.
int buffer_cache_store(id<MTLBuffer> buffer, const char **error) {
  if (buffer == nil) {
    logError(error, @"missing buffer to store");
    return 0;
  }

  int bufferId = 0;
  [bufferLock lock];
  if (nextBufferId == INT_MAX) {
    [bufferLock unlock];
    logError(error, @"buffer id space exhausted");
    return 0;
  }
  bufferId = nextBufferId++;
  bufferCache[@(bufferId)] = buffer;
  [bufferLock unlock];

  return bufferId;
}

// Retrieve a buffer from the cache by ID, or nil if not found.
id<MTLBuffer> buffer_cache_retrieve(int bufferId) {
  [bufferLock lock];
  id<MTLBuffer> buffer = bufferCache[@(bufferId)];
  [bufferLock unlock];

  return buffer;
}

// Remove a buffer from the cache by ID and return it, or nil on error.
id<MTLBuffer> buffer_cache_remove(int bufferId, const char **error) {
  [bufferLock lock];
  id<MTLBuffer> buffer = bufferCache[@(bufferId)];
  if (buffer == nil) {
    [bufferLock unlock];
    logError(error, [NSString stringWithFormat:@"invalid buffer id: %d", bufferId]);
    return nil;
  }
  [bufferCache removeObjectForKey:@(bufferId)];
  [bufferLock unlock];

  return buffer;
}

// Allocate a block of shared CPU/GPU memory large enough to hold the specified
// number of bytes. Writes the buffer's ID to the return value and its contents
// pointer to *contents. Returns 0 and sets an error on failure.
int buffer_new(size_t size, void **contents, const char **error) {
  // Wrap the body so autoreleased temporaries (boxed NSNumber keys via
  // buffer_cache_store, any error NSString) are released when this returns; the
  // cgo caller has no ambient pool to drain them. The MTLBuffer itself is kept
  // alive by the strong reference held in bufferCache, and *contents is a raw
  // pointer into its shared memory that the buffer's lifetime backs.
  @autoreleasepool {
    id<MTLBuffer> buffer =
        [metal_device() newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      logError(error, [NSString stringWithFormat:@"failed to create buffer with %zu bytes", size]);
      return 0;
    }

    int bufferId = buffer_cache_store(buffer, error);
    if (bufferId == 0) {
      // buffer_cache_store has already populated *error with a specific message.
      return 0;
    }

    *contents = [buffer contents];
    return bufferId;
  }
}

// Free a cached buffer. If any error is encountered relinquishing the memory,
// this sets an error message in error.
//
// The actual deallocation happens via ARC once the last strong reference
// (the cache entry, just removed) is gone. Any GPU work already in flight
// retains its own reference and finishes safely.
_Bool buffer_close(int bufferId, const char **error) {
  // Wrap the body so the boxed NSNumber keys and any error NSString created in
  // buffer_cache_remove are released when this returns; the cgo caller has no
  // ambient pool to drain them.
  @autoreleasepool {
    id<MTLBuffer> buffer = buffer_cache_remove(bufferId, error);
    if (buffer == nil) {
      return false;
    }

    return true;
  }
}
