// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cpu_timer.h"
#include "logging.h"
#include <iostream>
#include <time.h>

namespace ark {

// Measure current time in second.
double cpu_timer(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        return -1;
    }
    return (tspec.tv_nsec / 1.0e9) + tspec.tv_sec;
}

// Measure current time in nanosecond.
long cpu_ntimer(void)
{
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) == -1) {
        return -1;
    }
    return tspec.tv_nsec;
}

// Sleep in second.
int cpu_timer_sleep(double sec)
{
    struct timespec tspec;
    tspec.tv_sec = (time_t)sec;
    tspec.tv_nsec = (long)((sec - tspec.tv_sec) * 1.0e9);
    return nanosleep(&tspec, 0);
}

// Sleep in nanosecond.
int cpu_ntimer_sleep(long nsec)
{
    struct timespec tspec;
    tspec.tv_sec = 0;
    tspec.tv_nsec = nsec;
    return nanosleep(&tspec, 0);
}

} // namespace ark
