#include "stdlib.h"

#ifdef BUILD_FOR_ESP32
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#endif

void *kmalloc(int size)
{

#if defined(BUILD_FOR_ESP32)

#if defined(CONFIG_SPIRAM_SUPPORT)
    return heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    return malloc(size);
#endif

#else
    return malloc(size);
#endif
}