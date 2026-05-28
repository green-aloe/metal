#include "BufferCache.h"
#include "FunctionCache.h"
#include "Metal.h"
#include "MetalInternal.h"
#import <Metal/Metal.h>

static id<MTLDevice> device = nil;

// Initialize the default GPU. This should be called only once for the lifetime
// of the app. Returns false if no Metal device is available or the function
// cache (and its command queue) could not be created; in that case the package
// is unusable and callers should fall back to a non-Metal code path.
_Bool metal_init(void) {
  device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    return false;
  }

  if (!function_cache_init()) {
    device = nil;
    return false;
  }
  buffer_cache_init();

  return true;
}

// metal_device returns the shared MTLDevice initialized by metal_init.
id<MTLDevice> metal_device(void) {
  return device;
}
