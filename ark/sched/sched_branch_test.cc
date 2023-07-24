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
        for (int tile_id = 0; tile_id < 4; ++tile_id) {
            int sm_id = (tile_id / 4) % 5;
            int warp_id = tile_id % 4;
            sb.add(/*opseq_id*/ 0, /*tile_id*/ tile_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 1);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].tile_id_begin, 0);
        UNITTEST_EQ(branches[0].tile_id_last, 3);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 1);
    }

    {
        ark::SchedBranch sb;
        for (int tile_id = 0; tile_id < 12; ++tile_id) {
            int sm_id = (tile_id / 4) % 5;
            int warp_id = tile_id % 4;
            sb.add(/*opseq_id*/ 0, /*tile_id*/ tile_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].tile_id_begin, 0);
        UNITTEST_EQ(branches[0].tile_id_last, 11);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 1);
    }

    {
        ark::SchedBranch sb;
        for (int tile_id = 0; tile_id < 28; ++tile_id) {
            int sm_id = (tile_id / 4) % 5;
            int warp_id = tile_id % 4;
            sb.add(/*opseq_id*/ 0, /*tile_id*/ tile_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].tile_id_begin, 0);
        UNITTEST_EQ(branches[0].tile_id_last, 19);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 1);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 4);
        UNITTEST_EQ(branches[1].tile_id_begin, 20);
        UNITTEST_EQ(branches[1].tile_id_last, 27);
        UNITTEST_EQ(branches[1].tile_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_tile, 1);
    }

    {
        ark::SchedBranch sb;
        for (int tile_id = 0; tile_id < 7; ++tile_id) {
            int sm_id = tile_id % 5;
            sb.add(/*opseq_id*/ 0, /*tile_id*/ tile_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ 0, /*warp_id_end*/ 2);
        }

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].tile_id_begin, 0);
        UNITTEST_EQ(branches[0].tile_id_last, 4);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 2);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 2);
        UNITTEST_EQ(branches[1].tile_id_begin, 5);
        UNITTEST_EQ(branches[1].tile_id_last, 6);
        UNITTEST_EQ(branches[1].tile_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_tile, 2);
    }

    {
        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 3, /*sm_id*/ 2, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 5, /*sm_id*/ 3, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 2);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_id_begin, 2);
        UNITTEST_EQ(branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].tile_id_begin, 3);
        UNITTEST_EQ(branches[0].tile_id_last, 4);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 1);

        UNITTEST_EQ(branches[1].opseq_id, 0);
        UNITTEST_EQ(branches[1].sm_id_begin, 3);
        UNITTEST_EQ(branches[1].sm_id_end, 4);
        UNITTEST_EQ(branches[1].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_id_end, 3);
        UNITTEST_EQ(branches[1].tile_id_begin, 5);
        UNITTEST_EQ(branches[1].tile_id_last, 7);
        UNITTEST_EQ(branches[1].tile_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_tile, 1);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_multi_opseq()
{
    {
        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 0, /*sm_id*/ 0, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 0, /*sm_id*/ 0, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 1, /*sm_id*/ 0, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 1, /*sm_id*/ 0, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*tile_id*/ 2, /*sm_id*/ 1, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 2, /*sm_id*/ 1, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 3, /*sm_id*/ 1, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 3, /*sm_id*/ 1, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*tile_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 4, /*sm_id*/ 2, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 5, /*sm_id*/ 2, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 5, /*sm_id*/ 2, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*tile_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 6, /*sm_id*/ 3, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 7, /*sm_id*/ 3, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        sb.add(/*opseq_id*/ 0, /*tile_id*/ 8, /*sm_id*/ 4, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 8, /*sm_id*/ 4, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*tile_id*/ 9, /*sm_id*/ 4, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 1, /*tile_id*/ 9, /*sm_id*/ 4, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        std::vector<ark::Branch> branches = sb.get_branches();

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].tile_id_begin, 0);
        UNITTEST_EQ(branches[0].tile_id_last, 9);
        UNITTEST_EQ(branches[0].tile_id_diff, 1);
        UNITTEST_EQ(branches[0].num_warps_per_tile, 1);

        UNITTEST_EQ(branches[1].opseq_id, 1);
        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 5);
        UNITTEST_EQ(branches[1].warp_id_begin, 2);
        UNITTEST_EQ(branches[1].warp_id_end, 4);
        UNITTEST_EQ(branches[1].tile_id_begin, 0);
        UNITTEST_EQ(branches[1].tile_id_last, 9);
        UNITTEST_EQ(branches[1].tile_id_diff, 1);
        UNITTEST_EQ(branches[1].num_warps_per_tile, 1);
    }

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_branch_single_opseq);
    UNITTEST(test_sched_branch_multi_opseq);
    return 0;
}
