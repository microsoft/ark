# Environment Variables

- `ARK_ROOT` (Default: `/usr/local/ark`)

    The installation directory of ARK. For C++, defaults to `/usr/local/ark` when unset. For Python, defaults to the ARK Python module's path.

- `ARK_LOG_LEVEL` (Default: `INFO`; Options: `DEBUG`, `INFO`, `WARN`, `ERROR`)

    The log level of ARK. Use `DEBUG` for verbose debugging information and use `ERROR` for quiet execution.

- `ARK_TMP` (Default: `/tmp/ark`)

    A directory to store temporal files that ARK generates.

- `ARK_KEEP_TMP` (Default: `1`; Options: `0`, `1`)

    If set to `1`, do not remove temporal files in the `ARK_TMP` directory, vice versa.

- `ARK_HOSTFILE` (Default: `${ARK_ROOT}/hostfile`)

    Path to a hostfile. Need to set for multi-node execution. Ranks will be assigned in the order that hosts appear in the hostfile (`ARK_NUM_RANKS_PER_HOST` ranks per host).

- `ARK_NUM_RANKS_PER_HOST` (Default: `8`)

    The number of ranks that each host runs. The behavior is undefined if the total number of ranks is not a multiple of `ARK_NUM_RANKS_PER_HOST`.

- `ARK_DISABLE_IB` (Default: `0`; Options: `0`, `1`)

    If set to `1`, disable ibverbs networking (i.e., disable multi-node execution).

- `ARK_IGNORE_BINARY_CACHE` (Default: `1`; Options: `0`, `1`)

    If set to `1`, ignore the binary cache and force ARK to recompile binaries on each run.
