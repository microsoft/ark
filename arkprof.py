import ark
import sys

ark.init()
ark.Profiler(ark.Plan.from_file(sys.argv[1])).run(
    iter=1000, profile_processor_groups=False
)
