// go:build darwin
//  +build darwin

#include "Cache.h"
#import <Metal/Metal.h>

id<MTLDevice> device;

// Initialize the default GPU. This should be called only once for the lifetime
// of the app.
void metal_init() {
  // Get the default MTLDevice (each GPU is assigned its own device).
  device = MTLCreateSystemDefaultDevice();
  NSCAssert(device != nil, @"Failed to find default GPU");
  cache_init();
}