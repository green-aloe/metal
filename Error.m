// go:build darwin
//  +build darwin

#include "Error.h"

// Log an error to console and optionally set target to the error message if
// target is not nil. If target already contains a message, it is appended to
// the new message.
void logError(const char **target, NSString *format, ...) {
  NSString *message = [NSString stringWithFormat:@"%@", format];
  NSLog(@"%@", message);

  if (target != nil) {
    if (strlen(*target) > 0) {
      message = [NSString stringWithFormat:@"%@: %s", message, *target];
    }
    *target = [message UTF8String];
  }
}
