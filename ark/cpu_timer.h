// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_CPU_TIMER_H_
#define ARK_CPU_TIMER_H_

namespace ark {

// Measure current time in second.
double cpu_timer(void);
// Measure current time in nanosecond.
long cpu_ntimer(void);
// Sleep in second.
int cpu_timer_sleep(double sec);
// Sleep in nanosecond.
int cpu_ntimer_sleep(long nsec);

} // namespace ark

#endif // ARK_CPU_TIMER_H_
