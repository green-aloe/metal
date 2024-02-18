// go:build darwin
//  +build darwin

#include "error.h"

// Log an error to console and optionally set target to the error message.
void logError(const char **target, NSString *format, ...) {
  NSString *message = [NSString stringWithFormat:@"%@", format];
  NSLog(@"%@", message);

  if (target != nil) {
    *target = [message UTF8String];
  }
}
