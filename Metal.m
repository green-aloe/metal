#import "BufferCache.h"
#import "FunctionCache.h"
#import "Metal.h"
#import "MetalInternal.h"
#import <Metal/Metal.h>

// This package relies on ARC to manage the lifetimes of its Metal objects
// (libraries, pipelines, buffers). cgo compiles .m files under manual reference
// counting unless -fobjc-arc is passed (set via #cgo CFLAGS in the Go sources),
// so fail the build loudly rather than silently leak if that flag is ever lost.
#if !__has_feature(objc_arc)
#error "this package requires ARC; build the cgo sources with -fobjc-arc"
#endif

static id<MTLDevice> device = nil;

// Initialize the default GPU. This should be called only once for the lifetime
// of the app. Returns false if no Metal device is available or the function
// cache (and its command queue) could not be created; in that case the package
// is unusable and callers should fall back to a non-Metal code path.
_Bool metal_init(void) {
  // Wrap the body so any autoreleased temporaries from device/queue setup are
  // released when this returns; the cgo caller has no ambient pool to drain
  // them. device, the caches, and the command queue are all held by strong
  // static references, so they survive the pool drain.
  @autoreleasepool {
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
}

// metal_device returns the shared MTLDevice initialized by metal_init.
id<MTLDevice> metal_device(void) {
  return device;
}
