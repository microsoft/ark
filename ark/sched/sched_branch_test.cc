// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include "logging.h"
#include "sched_branch.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_branch_single_opseq()
{
    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 4; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 1);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].uop_id_last, 3);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 1);
    }

    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 12; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].uop_id_last, 11);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 1);
    }

    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 28; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].uop_id_last, 19);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 1);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 4);
        UNITTEST_EQ(branches[1].uop_id_begin, 20);
        UNITTEST_EQ(branches[1].uop_id_last, 27);
        UNITTEST_EQ(branches[1].uop_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_uop, 1);
    }

    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 7; ++uop_id) {
            int sm_id = uop_id % 5;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ 0, /*warp_id_end*/ 2);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].uop_id_last, 4);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 2);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 2);
        UNITTEST_EQ(branches[1].uop_id_begin, 5);
        UNITTEST_EQ(branches[1].uop_id_last, 6);
        UNITTEST_EQ(branches[1].uop_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_uop, 2);
    }

    {
        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 3, /*sm_id*/ 2, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 5, /*sm_id*/ 3, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 2);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_id_begin, 2);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].uop_id_begin, 3);
        UNITTEST_EQ(branches[0].uop_id_last, 4);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 1);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 3);
        UNITTEST_EQ(branches[1].sm_id_end, 4);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 3);
        UNITTEST_EQ(branches[1].uop_id_begin, 5);
        UNITTEST_EQ(branches[1].uop_id_last, 7);
        UNITTEST_EQ(branches[1].uop_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_uop, 1);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_multi_opseq()
{
    {
        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 0, /*sm_id*/ 0, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 0, /*sm_id*/ 0, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 1, /*sm_id*/ 0, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 1, /*sm_id*/ 0, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*uop_id*/ 2, /*sm_id*/ 1, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 2, /*sm_id*/ 1, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 3, /*sm_id*/ 1, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 3, /*sm_id*/ 1, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*uop_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 5, /*sm_id*/ 2, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 5, /*sm_id*/ 2, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*uop_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*uop_id*/ 8, /*sm_id*/ 4, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 8, /*sm_id*/ 4, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 9, /*sm_id*/ 4, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*uop_id*/ 9, /*sm_id*/ 4, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].uop_id_last, 9);
        UNITTEST_EQ(branches[0].uop_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_uop, 1);

        UNITTEST_EQ(branches[1].opseq_id, 1);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 5);
        UNITTEST_EQ(branches[1].warp_id_begin, 2);
        UNITTEST_EQ(branches[1].warp_id_end, 4);
        UNITTEST_EQ(branches[1].uop_id_begin, 0);
        UNITTEST_EQ(branches[1].uop_id_last, 9);
        UNITTEST_EQ(branches[1].uop_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_uop, 1);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_clear()
{
    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 4; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 1UL);

        branches.clear();

        UNITTEST_EQ(branches.size(), 0UL);
    }
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_branch_single_opseq);
    UNITTEST(test_sched_branch_multi_opseq);
    UNITTEST(test_sched_branch_clear);
    return 0;
}
