#ifndef HEADER_BUFFER_CACHE
#define HEADER_BUFFER_CACHE

#import <Metal/Metal.h>

void buffer_cache_init(void);
int buffer_cache_store(id<MTLBuffer> buffer, const char **error);
id<MTLBuffer> buffer_cache_retrieve(int bufferId);
id<MTLBuffer> buffer_cache_remove(int bufferId, const char **error);

#endif
