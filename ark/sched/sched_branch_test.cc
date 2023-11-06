// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched_branch.h"

#include "ark.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_sched_branch_single_opseq() {
    std::map<int, int> sm_id_to_smem_per_warp;

    {
        // Test:
        //   if (0 <= sm_id < 1 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * warp_id + 0;
        //   }

        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 4; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 1);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);
    }

    {
        // Test:
        //   if (0 <= sm_id < 3 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * (warp_id + 4 * sm_id) + 0;
        //   }

        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 12; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 1UL);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);
    }

    {
        // Test:
        //   if (0 <= sm_id < 5 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * (warp_id + 4 * sm_id) + 0;
        //   }
        //   if (0 <= sm_id < 2 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * (warp_id + 4 * sm_id) + 20;
        //   }

        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 28; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);

        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_begin,
                    20);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);
    }

    {
        // Test:
        //   if (0 <= sm_id < 5 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * (warp_id + 4 * sm_id) + 0;
        //   }
        //   if (0 <= sm_id < 2 && 0 <= warp_id < 4) {
        //     op = 0; uop = 1 * (warp_id + 4 * sm_id) + 20;
        //   }
        //   if (sm_id == 2 && 0 <= warp_id < 2) {
        //     op = 0; uop = 1 * (warp_id) + 28;
        //   }

        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 30; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 3UL);

        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);

        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_begin,
                    20);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    4);

        UNITTEST_EQ(branches[2].sm_id_begin, 2);
        UNITTEST_EQ(branches[2].sm_id_end, 3);
        UNITTEST_EQ(branches[2].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[2].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[2].warp_branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[2].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[2].warp_branches[0].branch_ops[0].uop_id_begin,
                    28);
        UNITTEST_EQ(branches[2].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[2].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    2);
    }

    {
        // Test:
        //   if (0 <= sm_id < 5 && 0 <= warp_id < 2) {
        //     op = 0; uop = 1 * sm_id + 0;
        //   }
        //   if (0 <= sm_id < 2 && 0 <= warp_id < 2) {
        //     op = 0; uop = 1 * sm_id + 5;
        //   }

        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 7; ++uop_id) {
            int sm_id = uop_id % 5;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ 0, /*warp_id_end*/ 2);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 2);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    1);

        UNITTEST_EQ(branches[1].sm_id_begin, 0);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_begin, 5);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 2);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    1);
    }

    {
        // Test:
        //   if (2 <= sm_id < 3 && 2 <= warp_id < 4) {
        //     op = 0; uop = 1 * ((warp_id - 2) + 2 * (sm_id - 2)) + 3;
        //   }
        //   if (3 <= sm_id < 4 && 0 <= warp_id < 3) {
        //     op = 0; uop = 1 * (warp_id + 3 * (sm_id - 3)) + 5;
        //   }

        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 0, /*sm_id*/ 2, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 1, /*sm_id*/ 2, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 2, /*sm_id*/ 3, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 3, /*sm_id*/ 3, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 4, /*sm_id*/ 3, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 2UL);

        UNITTEST_EQ(branches[0].sm_id_begin, 2);
        UNITTEST_EQ(branches[0].sm_id_end, 3);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 2);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    2);

        UNITTEST_EQ(branches[1].sm_id_begin, 3);
        UNITTEST_EQ(branches[1].sm_id_end, 4);
        UNITTEST_EQ(branches[1].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_end, 3);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_begin, 2);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    3);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_multi_opseq() {
    std::map<int, int> sm_id_to_smem_per_warp;

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

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 1UL);

        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 5);
        UNITTEST_EQ(branches[0].warp_branches.size(), 2UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 2);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    2);
        UNITTEST_EQ(branches[0].warp_branches[1].warp_id_begin, 2);
        UNITTEST_EQ(branches[0].warp_branches[1].warp_id_end, 4);
        UNITTEST_EQ(branches[0].warp_branches[1].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[1].branch_ops[0].opseq_id, 1);
        UNITTEST_EQ(branches[0].warp_branches[1].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[1].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[1].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[1].branch_ops[0].num_uops_per_sm,
                    2);
    }

    {
        // Test:
        //   if (sm_id == 0 && warp_id == 0) {
        //     op = 0; uop = 0;
        //   }
        //   if (sm_id == 1 && 0 <= warp_id < 4) {
        //     op = 0; uop = warp_id + 1;
        //   }
        //
        // However, due to the greedy nature of the algorithm, the result will
        // be:
        //   if (0 <= sm_id < 2 && warp_id == 0) {
        //     op = 0; uop = sm_id;
        //   }
        //   if (sm_id == 1 && 1 <= warp_id < 4) {
        //     op = 0; uop = warp_id + 1;
        //   }

        ark::SchedBranch sb;
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 0, /*sm_id*/ 0, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 1, /*sm_id*/ 1, /*warp_id_begin*/ 0,
               /*warp_id_end*/ 1);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 2, /*sm_id*/ 1, /*warp_id_begin*/ 1,
               /*warp_id_end*/ 2);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 3, /*sm_id*/ 1, /*warp_id_begin*/ 2,
               /*warp_id_end*/ 3);
        sb.add(/*opseq_id*/ 0, /*uop_id*/ 4, /*sm_id*/ 1, /*warp_id_begin*/ 3,
               /*warp_id_end*/ 4);

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 2UL);
        UNITTEST_EQ(branches[0].sm_id_begin, 0);
        UNITTEST_EQ(branches[0].sm_id_end, 2);
        UNITTEST_EQ(branches[0].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_begin, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].warp_id_end, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].uop_id_begin, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[0].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[0].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    1);
        UNITTEST_EQ(branches[1].sm_id_begin, 1);
        UNITTEST_EQ(branches[1].sm_id_end, 2);
        UNITTEST_EQ(branches[1].warp_branches.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_begin, 1);
        UNITTEST_EQ(branches[1].warp_branches[0].warp_id_end, 4);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops.size(), 1UL);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].opseq_id, 0);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_begin, 2);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].uop_id_diff, 1);
        UNITTEST_EQ(
            branches[1].warp_branches[0].branch_ops[0].num_warps_per_uop, 1);
        UNITTEST_EQ(branches[1].warp_branches[0].branch_ops[0].num_uops_per_sm,
                    3);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_clear() {
    std::map<int, int> sm_id_to_smem_per_warp;
    {
        ark::SchedBranch sb;
        for (int uop_id = 0; uop_id < 4; ++uop_id) {
            int sm_id = (uop_id / 4) % 5;
            int warp_id = uop_id % 4;
            sb.add(/*opseq_id*/ 0, /*uop_id*/ uop_id, /*sm_id*/ sm_id,
                   /*warp_id_begin*/ warp_id, /*warp_id_end*/ warp_id + 1);
        }

        std::vector<ark::Branch> branches =
            sb.get_branches(sm_id_to_smem_per_warp);

        UNITTEST_EQ(branches.size(), 1UL);

        branches.clear();

        UNITTEST_EQ(branches.size(), 0UL);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_branch_errors() {
    {
        ark::SchedBranch sb;
        // negative uop_id
        UNITTEST_THROW(sb.add(0, -1, 0, 0, 1), ark::SchedulerError);
        // negative sm_id
        UNITTEST_THROW(sb.add(0, 0, -1, 0, 1), ark::SchedulerError);
        // negative warp_id_begin
        UNITTEST_THROW(sb.add(0, 0, 0, -1, 1), ark::SchedulerError);
        // negative warp_id_end
        UNITTEST_THROW(sb.add(0, 0, 0, 0, -1), ark::SchedulerError);
        // warp_id_begin >= warp_id_end
        UNITTEST_THROW(sb.add(0, 0, 0, 1, 1), ark::SchedulerError);

        sb.add(0, 0, 0, 0, 1);
        // duplicate uop_id
        UNITTEST_THROW(sb.add(0, 0, 0, 0, 1), ark::SchedulerError);

        sb.add(0, 1, 0, 0, 1);
        // duplicate uop_id
        UNITTEST_THROW(sb.add(0, 1, 0, 0, 1), ark::SchedulerError);
    }
    {
        // Different number of warps per uop
        ark::SchedBranch sb;
        sb.add(0, 0, 0, 0, 1);
        sb.add(0, 1, 0, 1, 3);
        UNITTEST_THROW(sb.get_branches({}), ark::SchedulerError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_sched_branch_single_opseq);
    UNITTEST(test_sched_branch_multi_opseq);
    UNITTEST(test_sched_branch_clear);
    UNITTEST(test_sched_branch_errors);
    return 0;
}
