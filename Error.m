#include "Error.h"
#include <string.h>

// Set *target to a heap-allocated copy of message's UTF-8 bytes if target is
// not nil. The caller (Go side) is responsible for free()ing *target.
//
// -[NSString UTF8String] returns a pointer whose lifetime is tied to the
// receiver's autorelease pool; that pool can drain at any cgo call boundary,
// so we strdup the bytes before handing them back across the language barrier.
void logError(const char **target, NSString *message) {
  if (target == nil) {
    return;
  }

  const char *bytes = [message UTF8String];
  if (bytes == NULL) {
    *target = NULL;
    return;
  }

  *target = strdup(bytes);
}
