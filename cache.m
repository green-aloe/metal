// go:build darwin
//  +build darwin

#include "error.h"
#import <Metal/Metal.h>

void **cache = nil;
int numItems = 0;

// The @synchronized directive requires an Objective-c object. We'll use this
// (even though we're not using the locking functionality) because it won't ever
// be touched for anything else.
NSLock *cacheLock = nil;

// Initialize the global cache.
void cache_init() { cacheLock = [[NSLock alloc] init]; }

// Add an item to the cache. This returns 0 and logs a message if any error is
// encountered and the item is not cached.
int cache_cache(void *item, const char **error) {
  if (item == nil) {
    logError(error, @"Missing item to cache");
    return 0;
  }

  int cacheId = 0;

  @synchronized(cacheLock) {
    numItems++;
    cache = realloc(cache, sizeof(void *) * numItems);
    if (cache == nil) {
      logError(error, @"Failed to grow cache");
      return 0;
    }

    cache[numItems - 1] = item;

    // A cache Id is an item's 1-based index in the cache.
    cacheId = numItems;
  }

  return cacheId;
}

// Retrieve an item from the cache. This returns nil and logs a message if any
// error is encountered and the item is not retrieved.
void *cache_retrieve(int cacheId, const char **error) {
  if (cacheId < 1) {
    logError(error,
             [NSString stringWithFormat:@"Invalid cache Id: %d", cacheId]);
    return nil;
  }

  @synchronized(cacheLock) {
    if (cacheId > numItems) {
      logError(error,
               [NSString stringWithFormat:@"Invalid cache Id: %d", cacheId]);
      return nil;
    }

    // A cache Id is an item's 1-based index in the cache. We need to convert
    // it into a 0-based index to retrieve it from the cache.
    int index = cacheId - 1;

    return cache[index];
  }
}

// Remove an item from the cache.
_Bool cache_remove(int cacheId, const char **error) {
  if (cacheId < 1) {
    logError(error,
             [NSString stringWithFormat:@"Invalid cache Id: %d", cacheId]);
    return false;
  }

  @synchronized(cacheLock) {
    if (cacheId > numItems) {
      logError(error,
               [NSString stringWithFormat:@"Invalid cache Id: %d", cacheId]);
      return false;
    }

    // A cache Id is an item's 1-based index in the cache. We need to convert
    // it into a 0-based index to retrieve it from the cache.
    int index = cacheId - 1;

    // Set the item to nil.
    cache[index] = nil;
  }

  return true;
}