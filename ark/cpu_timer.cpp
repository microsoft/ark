// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cpu_timer.h"

#include <time.h>

namespace ark {

// Measure current time in second.
double cpu_timer(void) {
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        return -1;
    }
    return (tspec.tv_nsec / 1.0e9) + tspec.tv_sec;
}

}  // namespace ark
