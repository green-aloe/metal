// go:build darwin
//  +build darwin

#ifndef HEADER_ERROR
#define HEADER_ERROR

#import <Metal/Metal.h>

void logError(const char **, NSString *, ...);

#endif