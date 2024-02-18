// go:build darwin
//  +build darwin

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
int cache_cache(void *item) {
  if (item == nil) {
    NSLog(@"Missing item to cache");
    return 0;
  }

  int cacheId = 0;

  @synchronized(cacheLock) {
    numItems++;
    cache = realloc(cache, sizeof(void *) * numItems);
    if (cache == nil) {
      NSLog(@"Failed to grow cache");
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
void *cache_retrieve(int cacheId) {
  void *item = nil;

  if (cacheId < 1) {
    NSLog(@"Invalid cache Id: %d", cacheId);
    return nil;
  }

  @synchronized(cacheLock) {
    if (cacheId > numItems) {
      NSLog(@"Invalid cache Id: %d", cacheId);
      return nil;
    }

    // A cache Id is an item's 1-based index in the cache. We need to convert
    // it into a 0-based index to retrieve it from the cache.
    int index = cacheId - 1;

    item = cache[index];
  }

  return item;
}

// Remove an item from the cache.
void cache_remove(int cacheId) {
  if (cacheId < 1) {
    NSLog(@"Invalid cache Id: %d", cacheId);
    return;
  }

  @synchronized(cacheLock) {
    if (cacheId > numItems) {
      NSLog(@"Invalid cache Id: %d", cacheId);
      return;
    }

    // A cache Id is an item's 1-based index in the cache. We need to convert
    // it into a 0-based index to retrieve it from the cache.
    int index = cacheId - 1;

    // Set the item to nil.
    cache[index] = nil;
  }
}