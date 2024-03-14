// go:build darwin
//  +build darwin

#ifndef HEADER_CACHE
#define HEADER_CACHE

void cache_init();

int cache_cache(void *item, **error);
void *cache_retrieve(int cacheId, **error);
void cache_remove(int cacheId, **error);

#endif