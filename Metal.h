#ifndef HEADER_METAL
#define HEADER_METAL

#include <stdlib.h>

// Functions that must be called once for every application
_Bool metal_init(void);

// Functions that must be called once for every metal function
int function_new(const char *metalCode, const char *funcName,
                 const char **error);
_Bool function_run(int functionId, unsigned int width, unsigned int height,
                   unsigned int depth, float *inputs, int numInputs,
                   int *bufferIds, int numBufferIds, const char **error,
                   int *errorCode);
_Bool function_run_batch(int numDispatches, int *functionIds,
                         unsigned int *widths, unsigned int *heights,
                         unsigned int *depths, float **inputs, int *numInputs,
                         int **bufferIds, int *numBufferIds,
                         const char **error, int *errorCode);
_Bool function_run_async(int functionId, unsigned int width,
                         unsigned int height, unsigned int depth, float *inputs,
                         int numInputs, int *bufferIds, int numBufferIds,
                         void **handle, const char **error, int *errorCode);
_Bool function_run_batch_async(int numDispatches, int *functionIds,
                               unsigned int *widths, unsigned int *heights,
                               unsigned int *depths, float **inputs,
                               int *numInputs, int **bufferIds,
                               int *numBufferIds, void **handle,
                               const char **error, int *errorCode);
_Bool function_wait(void *handle, const char **error);

// Functions for querying data on a metal function
const char *function_name(int functionId);

// Functions that must be called once for every buffer used as an argument to
// a metal function
int buffer_new(size_t size, void **contents, const char **error);
_Bool buffer_close(int bufferId, const char **error, int *errorCode);

// Functions for closing metal resources
_Bool function_close(int functionId, const char **error, int *errorCode);

#endif