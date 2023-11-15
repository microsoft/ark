// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched.h"

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "logging.h"
#include "ops/ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;

ark::unittest::State test_sched_mm_add() {
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
        ark::DefaultScheduler sched{model, 0, 0, 1, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sched_mm_add");
        sched.schedule();
        auto codes = sched.gen_code();

        GpuLoopKernel glk{"test_sched_mm_add",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx};
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        int iter = 1000;
        glk.run(iter);
        glk.stop();

        UNITTEST_LOG("test_sched_mm_add: batch_size ", batch_size,
                     " dim_input ", dim_input, " dim_hidden ", dim_hidden,
                     " dtype ", dtype, " elapsed ",
                     glk.get_elapsed_msec() / (float)iter, " ms/iter");

        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_scheduler_simple_mm() {
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

    DefaultScheduler sched{m, 0, 0, 1, 8};
    GpuMgrCtx *ctx = sched.create_context("test_scheduler_simple_mm");
    sched.schedule();
    auto codes = sched.gen_code();

    GpuLoopKernel glk{"test_scheduler_simple_mm",
                      codes,
                      (unsigned int)ginfo.num_sm,
                      8,
                      (unsigned int)ginfo.smem_block_total,
                      "",
                      ctx};
    glk.compile(ginfo);
    glk.load();

    GpuStream stream = ctx->create_stream();
    for (int i = 0; i < 10; ++i) {
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        glk.run(100);
        glk.stop();
        UNITTEST_LOG(glk.get_elapsed_msec());
    }

    return unittest::SUCCESS;
}

Tensor *MultiheadAttention(Model *model, Tensor *input, DimType embed_dim,
                           DimType num_heads, float dropout, TensorType dtype) {
    // input: (batch_size, seq_len, embed_dim)
    // output: (batch_size, seq_len, embed_dim)
    Tensor *w_q_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_k_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_v_proj = model->tensor({embed_dim, embed_dim}, dtype);
    Tensor *w_out_proj = model->tensor({embed_dim, embed_dim}, dtype);

    Tensor *q_proj = model->matmul(input, w_q_proj);
    Tensor *k_proj = model->matmul(input, w_k_proj);
    Tensor *v_proj = model->matmul(input, w_v_proj);
    Tensor *q_proj_r_t = model->reshape(
        q_proj,
        {input->shape[0], input->shape[1], num_heads, embed_dim / num_heads});
    Tensor *k_proj_r_t = model->reshape(
        k_proj,
        {input->shape[0], input->shape[1], num_heads, embed_dim / num_heads});
    Tensor *v_proj_r_t = model->reshape(
        v_proj,
        {input->shape[0], input->shape[1], num_heads, embed_dim / num_heads});
    // Tensor *q_proj_r_t = model->transpose(q_proj_r, {0, 2, 1, 3});
    // Tensor *k_proj_r_t = model->transpose(k_proj_r, {0, 2, 1, 3});
    // Tensor *v_proj_r_t = model->transpose(v_proj_r, {0, 2, 1, 3});
    q_proj_r_t = model->reshape(
        q_proj_r_t,
        {input->shape[0] * num_heads, input->shape[1], embed_dim / num_heads});
    k_proj_r_t = model->reshape(
        k_proj_r_t,
        {input->shape[0] * num_heads, input->shape[1], embed_dim / num_heads});
    v_proj_r_t = model->reshape(
        v_proj_r_t,
        {input->shape[0] * num_heads, input->shape[1], embed_dim / num_heads});

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
                                float dropout, TensorType dtype) {
    Tensor *attn_out =
        MultiheadAttention(model, input, embed_dim, num_heads, dropout, dtype);
    Tensor *res = model->add(input, attn_out);
    // res = model->layernorm(res, res);

    Tensor *w_ff1 = model->tensor({embed_dim, dim_ff}, dtype);
    Tensor *w_ff2 = model->tensor({dim_ff, embed_dim}, dtype);

    Tensor *ff1 = model->matmul(res, w_ff1);
    Tensor *ff2 = model->matmul(ff1, w_ff2);
    Tensor *ret = model->add(res, ff2);
    // ret = model->layernorm(ret, ret);
    return ret;
}

Tensor *GPT3LayerForward(Model *model, Tensor *input, TensorType dtype) {
    return TransformerLayerForward(model, input,
                                   /*embed_dim=*/12288,
                                   /*num_heads=*/96,
                                   /*dim_ff=*/49152,
                                   /*dropout=*/0.0, dtype);
}

ark::unittest::State test_sched_gpt3() {
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
        ark::DefaultScheduler sched{model, 0, 0, 1, 8};
        GpuMgrCtx *ctx = sched.create_context("test_sched_gpt3");
        sched.schedule();
        auto codes = sched.gen_code();

        GpuLoopKernel glk{"test_sched_gpt3",
                          codes,
                          (unsigned int)ginfo.num_sm,
                          8,
                          (unsigned int)ginfo.smem_block_total,
                          "",
                          ctx};
        glk.compile(ginfo);
        glk.load();
        GpuStream stream = ctx->create_stream();
        GpuState ret = glk.launch(stream, false);
        UNITTEST_EQ(ret, 0);
        int iter = 100;
        glk.run(iter);
        glk.stop();

        UNITTEST_LOG("test_sched_gpt3: batch_size ", batch_size, " seq_len ",
                     seq_len, " embed_dim ", embed_dim, " dtype ", dtype,
                     " elapsed ", glk.get_elapsed_msec() / (float)iter,
                     " ms/iter");

        return unittest::SUCCESS;
    });
    ark::unittest::wait_all_processes();
    return unittest::SUCCESS;
}

ark::unittest::State test_sched_many_comm_ops() {
    constexpr int num_gpus = 4;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        ark::unittest::spawn_process([gpu_id, num_gpus]() {
            // Each GPU's data is equal to its GPU ID + 1.
            ark::Model m{gpu_id};

            for (int i = 0; i < 100; ++i) {
                ark::Tensor *data = m.tensor(ark::Dims(4096), ark::FP16);
                m.all_gather(data, gpu_id, num_gpus);
            }

            ark::Executor exe{gpu_id, num_gpus, m, "test_sched_many_comm_ops"};
            exe.compile();
            exe.launch();
            exe.run(3);
            exe.stop();
            return ark::unittest::SUCCESS;
        });
    }
    ark::unittest::wait_all_processes();
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_mixed_precision() {
    ark::Model m;
    ark::Tensor *x0 = m.tensor({2, 128, 128}, ark::FP16);
    ark::Tensor *x1 = m.scale(x0, 0.7);
    ark::Tensor *x2 = m.cast(x1, ark::FP32);
    ark::Tensor *x3 = m.tensor({2, 128, 128}, ark::FP32);
    m.matmul(x2, x3);

    ark::Executor exe{0, 1, m, "sched_mixed_precision"};
    exe.compile();
    exe.launch();
    exe.run(3);
    exe.stop();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_parallel_matmul() {
    ark::Model m;
    ark::Tensor *t0 = m.tensor({256, 8192}, ark::FP16);
    ark::Tensor *t1 = m.tensor({8192, 8192}, ark::FP16);
    auto shards = m.sharding(t0, 0, 128);

    m.matmul(t0, t1);
    m.matmul(shards[0], t1);

    ark::Executor exe{0, 1, m, "sched_parallel_matmul"};
    exe.compile();
    exe.launch();
    exe.run(3);
    exe.stop();

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sched_graph_opt() {
    ark::Model m;
    ark::Tensor *ones = m.tensor({128, 8192}, ark::FP32);
    ark::Tensor *ppp_ones = m.scale(ones, 0.001);
    ark::Tensor *w = m.tensor({8192, 256}, ark::FP32);

    ark::Tensor *y = m.matmul(ppp_ones, w);
    ark::Tensor *ones2 = m.tensor({128, 256}, ark::FP32);
    ark::Tensor *y_plus_one = m.add(y, ones2);

    ark::Executor exe{0, 1, m, "sched_graph_opt"};
    exe.compile();

    std::vector<float> ones_data(ones->shape.size(), 1.0f);
    std::vector<float> ones2_data(ones2->shape.size(), 1.0f);
    std::vector<float> w_data(w->shape.size(), 1.0f);
    ones->write(ones_data.data());
    ones2->write(ones2_data.data());
    w->write(w_data.data());

    exe.launch();
    exe.run(1);
    exe.stop();

    std::vector<float> output_y(y->shape.size());
    y->read(output_y.data());

    for (float v : output_y) {
        UNITTEST_EQ(int(v * 100), 819);
    }

    std::vector<float> output_y_plus_one(y_plus_one->shape.size());
    y_plus_one->read(output_y_plus_one.data());

    for (float v : output_y_plus_one) {
        UNITTEST_EQ(int(v * 100), 919);
    }

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    // UNITTEST(test_sched_mm_add);
    // UNITTEST(test_scheduler_simple_mm);
    // UNITTEST(test_sched_gpt3);
    UNITTEST(test_sched_many_comm_ops);
    UNITTEST(test_sched_mixed_precision);
    UNITTEST(test_sched_parallel_matmul);
    // UNITTEST(test_sched_graph_opt);
    return 0;
}
