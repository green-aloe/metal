// Internal header for the Objective-C / Metal sources in this package.
// Not included by Go via cgo (uses Objective-C types).

#ifndef HEADER_METAL_INTERNAL
#define HEADER_METAL_INTERNAL

#import <Metal/Metal.h>

// metal_device returns the shared MTLDevice initialized by metal_init.
// Returns nil if metal_init has not been called.
id<MTLDevice> metal_device(void);

#endif
