// go:build darwin
//  +build darwin

#ifndef HEADER_METAL
#define HEADER_METAL

#include <stdlib.h>

// Functions that must be called once for every application
void metal_init();

// Functions that must be called once for every metal function
int function_new(const char *metalCode, const char *funcName, const char **);
_Bool function_run(int functionId, int width, int height, int depth,
                   int *bufferIds, int numBufferIds, const char **);

// Functions for querying data on a metal function
const char *function_name(int);

// Functions that must be called once for every buffer used as an argument to
// a metal function
int buffer_new(int size, const char **);
void *buffer_retrieve(int bufferId, const char **);

#endif