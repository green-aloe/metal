#ifndef HEADER_ERROR
#define HEADER_ERROR

#import <Metal/Metal.h>

// MetalErrorCode categorizes a failure so the Go layer can map it to a sentinel
// (errors.Is) without parsing the human-readable message. The message and the
// code are independent: the message is for display, the code is for matching.
enum MetalErrorCode {
  MetalErrorNone = 0,
  MetalErrorInvalidFunctionId = 1,
  MetalErrorInvalidBufferId = 2,
};

// logError writes a heap-allocated copy of message to *target (for display).
void logError(const char **target, NSString *message);

// setErrorCode writes code to *target. Both logError and setErrorCode are
// nil-safe in their out-param, so a caller that does not care about one can
// pass NULL for it.
void setErrorCode(int *target, enum MetalErrorCode code);

#endif