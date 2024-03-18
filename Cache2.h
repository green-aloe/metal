// go:build darwin
//  +build darwin

#ifndef HEADER_CACHE
#define HEADER_CACHE

void cache_init();

int cache_cache(void *item, const char **error);
void *cache_retrieve(int cacheId, const char **error);
_Bool cache_remove(int cacheId, const char **error);

#endif