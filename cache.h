// go:build darwin
//  +build darwin

#ifndef HEADER_CACHE
#define HEADER_CACHE

void cache_init();

int cache_cache(void *item);
void *cache_retrieve(int cacheId);
void cache_remove(int cacheId);

#endif