#import "Error.h"
#include <string.h>

// Set *target to a heap-allocated copy of message's UTF-8 bytes if target is
// not nil. The caller (Go side) is responsible for free()ing *target.
//
// -[NSString UTF8String] returns a pointer whose lifetime is tied to the
// receiver's autorelease pool; that pool can drain at any cgo call boundary,
// so we strdup the bytes before handing them back across the language barrier.
void logError(const char **target, NSString *message) {
  if (target == NULL) {
    return;
  }

  const char *bytes = [message UTF8String];
  if (bytes == NULL) {
    *target = NULL;
    return;
  }

  *target = strdup(bytes);
}

// Set *target to code if target is not NULL. Callers that report a plain
// (uncategorized) error leave the code at its zero value, MetalErrorNone.
void setErrorCode(int *target, enum MetalErrorCode code) {
  if (target == NULL) {
    return;
  }

  *target = (int)code;
}
