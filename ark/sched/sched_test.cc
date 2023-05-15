// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/sched/sched.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;
using namespace ark;

ark::unittest::State test_sched_mm_add()
{
    unittest::spawn_process([&]() {
        DimType batch_size = 1;
        DimType dim_input = 2048;
        DimType dim_hidden = 12288;
        TensorType dtype = FP16;

        Model model;
        Tensor *input =
            model.tensor({batch_size, dim_input, dim_hidden}, dtype);
        Tensor *weight = model.tensor({dim_hidden, dim_hidden}, dtype);

        Tensor *mm = model.matmul(input, weight);
        /* Tensor *mm_add = */ model.add(mm, input);

        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::DefaultScheduler sched{0, 0, 1, model, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sched_mm_add");
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sched_mm_add",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        int iter = 1000;
        glk.run(iter);
        glk.stop();

        LOG(INFO, "test_sched_mm_add: batch_size ", batch_size, " dim_input ",
            dim_input, " dim_hidden ", dim_hidden, " dtype ", dtype,
            " elapsed ", glk.get_elapsed_msec() / (float)iter, " ms/iter");

        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_scheduler_simple_mm()
{
    // Hidden dimension of the dense layer.
    unsigned int units = 2048;
    // Input dimension of the dense layer.
    unsigned int in_dim = 2048;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 2048;
    // Batch size of the input.
    unsigned int batch_size = 1;

    Model m;
    Tensor *input = m.tensor({batch_size, channel, in_dim}, FP16);
    Tensor *weight = m.tensor({in_dim, units}, FP16);
    m.matmul(input, weight);

    GpuMgr *mgr = get_gpu_mgr(0);
    const GpuInfo &ginfo = mgr->get_gpu_info();

    DefaultScheduler sched{0, 0, 1, m, 8};
    GpuMgrCtx *ctx = sched.create_context("test_scheduler_simple_mm");
    auto codes = sched.schedule();
    unsigned int num_depths = sched.get_num_depths();

    GpuLoopKernel glk{"test_scheduler_simple_mm",
                      codes,
                      (unsigned int)ginfo.num_sm,
                      8,
                      (unsigned int)ginfo.smem_block_total,
                      "",
                      ctx,
                      num_depths};
    glk.compile(ginfo);
    glk.load();

    GpuStream stream = ctx->create_stream();
    for (int i = 0; i < 10; ++i) {
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(100);
        glk.stop();
        LOG(INFO, glk.get_elapsed_msec());
    }

    return unittest::SUCCESS;
}

Tensor *MultiheadAttention(Model *model, Tensor *input, DimType embed_dim,
                           DimType num_heads, float dropout, TensorType dtype)
{
    // input: (batch_size, seq_len, embed_dim)
    // output: (batch_size, seq_len, embed_dim)
    Tensor *w_q_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_k_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_v_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_out_proj = model->tensor({embed_dim, embed_dim}, dtype);

    Tensor *q_proj = model->matmul(input, w_q_proj);
    Tensor *k_proj = model->matmul(input, w_k_proj);
    Tensor *v_proj = model->matmul(input, w_v_proj);
    Tensor *q_proj_r_t =
        model->reshape(q_proj, {input->shape[0], input->shape[1], num_heads,
                                embed_dim / num_heads});
    Tensor *k_proj_r_t =
        model->reshape(k_proj, {input->shape[0], input->shape[1], num_heads,
                                embed_dim / num_heads});
    Tensor *v_proj_r_t =
        model->reshape(v_proj, {input->shape[0], input->shape[1], num_heads,
                                embed_dim / num_heads});
    // Tensor *q_proj_r_t = model->transpose(q_proj_r, {0, 2, 1, 3});
    // Tensor *k_proj_r_t = model->transpose(k_proj_r, {0, 2, 1, 3});
    // Tensor *v_proj_r_t = model->transpose(v_proj_r, {0, 2, 1, 3});
    q_proj_r_t =
        model->reshape(q_proj_r_t, {input->shape[0] * num_heads,
                                    input->shape[1], embed_dim / num_heads});
    k_proj_r_t =
        model->reshape(k_proj_r_t, {input->shape[0] * num_heads,
                                    input->shape[1], embed_dim / num_heads});
    v_proj_r_t =
        model->reshape(v_proj_r_t, {input->shape[0] * num_heads,
                                    input->shape[1], embed_dim / num_heads});

    // scaled dot product
    Tensor *attn_logits =
        model->matmul(q_proj_r_t, k_proj_r_t, nullptr, 1, false, true);
    Tensor *attn_logits_scaled =
        model->scale(attn_logits, 1.0 / sqrt(embed_dim / num_heads));

    // Tensor *attention = model->softmax(attn_logits_scaled, 2);
    Tensor *attention = attn_logits_scaled;
    Tensor *values = model->matmul(attention, v_proj_r_t);
    // values = model->reshape(values, {input->shape[0], num_heads,
    // input->shape[1], embed_dim / num_heads});

    // Tensor *values_t = model->transpose(values, {0, 2, 1, 3});
    Tensor *values_t_r =
        model->reshape(values, {input->shape[0], input->shape[1], embed_dim});
    Tensor *output = model->matmul(values_t_r, w_out_proj);

    if (dropout > 0.0) {
        // output = model->dropout(output, dropout);
    }
    return output;
}

Tensor *TransformerLayerForward(Model *model, Tensor *input, DimType embed_dim,
                                DimType num_heads, DimType dim_ff,
                                float dropout, TensorType dtype)
{
    Tensor *attn_out =
        MultiheadAttention(model, input, embed_dim, num_heads, dropout, dtype);
    Tensor *res = model->add(input, attn_out);
    // res = model->layer_norm(res, res);

    Tensor *w_ff1 = model->tensor({embed_dim, dim_ff}, dtype);
    Tensor *w_ff2 = model->tensor({dim_ff, embed_dim}, dtype);

    Tensor *ff1 = model->matmul(res, w_ff1);
    Tensor *ff2 = model->matmul(ff1, w_ff2);
    Tensor *ret = model->add(res, ff2);
    // ret = model->layer_norm(ret, ret);
    return ret;
}

Tensor *GPT3LayerForward(Model *model, Tensor *input, TensorType dtype)
{
    return TransformerLayerForward(model, input,
                                   /*embed_dim=*/12288,
                                   /*num_heads=*/96,
                                   /*dim_ff=*/49152,
                                   /*dropout=*/0.0, dtype);
}

ark::unittest::State test_sched_gpt3()
{
    Model model;
    DimType batch_size = 1;
    DimType seq_len = 2048;
    DimType embed_dim = 12288;
    TensorType dtype = FP16;
    Tensor *input = model.tensor({batch_size, seq_len, embed_dim}, dtype);
    GPT3LayerForward(&model, input, dtype);

    unittest::spawn_process([&]() {
        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::DefaultScheduler sched{0, 0, 1, model, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sched_gpt3");
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_sched_gpt3",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        int iter = 100;
        glk.run(iter);
        glk.stop();

        LOG(INFO, "test_sched_gpt3: batch_size ", batch_size, " seq_len ",
            seq_len, " embed_dim ", embed_dim, " dtype ", dtype, " elapsed ",
            glk.get_elapsed_msec() / (float)iter, " ms/iter");

        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_sched_comp_baseline()
{
    // Hidden dimension of the dense layer.
    unsigned int units = 512;
    // Input dimension of the dense layer.
    unsigned int in_dim = 512;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 512;
    // Batch size of the input.
    unsigned int batch_size = 1;
    int bytes = channel * in_dim * sizeof(half_t);
    int input_tensor_num = 3;
    ark::srand();
    vector<unique_ptr<half_t[]>> input_data(input_tensor_num);
    for (int i = 0; i < input_tensor_num; i++) {
        input_data[i] = rand_halfs(channel * in_dim, 0.01);
    }
    // the result of the new scheduler
    half_t *output_data1 = (half_t *)malloc(bytes);
    UNITTEST_NE(output_data1, (void *)nullptr);

    // the result of the old scheduler
    half_t *output_data2 = (half_t *)malloc(bytes);
    UNITTEST_NE(output_data2, (void *)nullptr);

    // test the baseline scheduler
    ark::unittest::spawn_process([&]() {
        Model m;
        Tensor *input[3];
        input[0] = m.tensor({batch_size, channel, in_dim}, FP16);
        input[1] = m.tensor({batch_size, in_dim, units}, FP16);
        input[2] = m.tensor({batch_size, units, units}, FP16);

        Tensor *middle_result = m.matmul(input[0], input[1]);

        Tensor *middle_result1 = m.add(middle_result, input[2]);
        Tensor *output = m.scale(middle_result1, 2.3);
        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::SimpleScheduler sched{0, 0, 1, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_scheduler_simple_mm");
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_scheduler_simple_mm",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        glk.compile(ginfo);
        for (int i = 0; i < input_tensor_num; i++) {
            // Get the auto-scheduled buffers.
            ark::GpuBuf *input_tensor_buf = sched.get_gpu_buf(input[i]);
            UNITTEST_NE(input_tensor_buf, (ark::GpuBuf *)nullptr);

            // Set data.
            ark::gpu_memcpy(input_tensor_buf, input_data[i].get(), bytes);
        }
        // load the data into the input

        glk.load();

        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(1);
        glk.stop();
        ark::GpuBuf *output_tensor_buf = sched.get_gpu_buf(output);
        ark::gpu_memcpy(output_data1, output_tensor_buf, bytes);
        for (int i = 0; i < 10; i++) {
            LOG(DEBUG, "output_data1: ", (float)output_data1[i]);
        }
        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();

    // test the old scheduler
    ark::unittest::spawn_process([&]() {
        Model m;
        Tensor *input[3];
        input[0] = m.tensor({batch_size, channel, in_dim}, FP16);
        input[1] = m.tensor({batch_size, in_dim, units}, FP16);
        input[2] = m.tensor({batch_size, units, units}, FP16);

        Tensor *middle_result = m.matmul(input[0], input[1]);

        Tensor *middle_result1 = m.add(middle_result, input[2]);
        Tensor *output = m.scale(middle_result1, 2.3);

        GpuMgr *mgr = get_gpu_mgr(0);
        const GpuInfo &ginfo = mgr->get_gpu_info();
        ark::DefaultScheduler sched{0, 0, 1, m, 8};
        GpuMgrCtx *ctx = sched.create_context("test_scheduler_simple_mm");
        auto codes = sched.schedule();
        unsigned int num_depths = sched.get_num_depths();

        GpuLoopKernel glk{"test_scheduler_simple_mm",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx,
                          num_depths};
        glk.compile(ginfo);
        for (int i = 0; i < input_tensor_num; i++) {
            // Get the auto-scheduled buffers.
            ark::GpuBuf *input_tensor_buf = sched.get_gpu_buf(input[i]);
            UNITTEST_NE(input_tensor_buf, (ark::GpuBuf *)nullptr);

            // Set data.
            ark::gpu_memcpy(input_tensor_buf, input_data[i].get(), bytes);
        }
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(1);
        glk.stop();
        ark::GpuBuf *output_tensor_buf = sched.get_gpu_buf(output);
        ark::gpu_memcpy(output_data2, output_tensor_buf, bytes);
        for (int i = 0; i < 10; i++) {
            LOG(DEBUG, "output_data2: ", (float)output_data2[i]);
        }
        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    // TODO: the output data are set on different processes,  we need to copy
    //  run the test on the same process
    auto p = cmp_matrix((half_t *)output_data1, (half_t *)output_data2, channel,
                        units);
    LOG(ark::INFO, " scheduler compare test: ", " total_bytes: ", bytes,
        " iter: ", 1, setprecision(4), " mse: ", p.first,
        " max_err: ", p.second * 100, "%");
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_sched_mm_add);
    // UNITTEST(test_scheduler_simple_mm);
    UNITTEST(test_sched_gpt3);
    UNITTEST(test_sched_comp_baseline);
    return 0;
}
