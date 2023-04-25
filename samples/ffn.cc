
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "ark/executor.h"
#include "ark/logging.h"
#include "ark/model_io.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/process.h"

using namespace std;
using namespace ark;

void print_tensor(Tensor *tensor, Executor *exe)
{
    cout << "tensor: " << tensor->name << endl;
    ark::GpuBuf *buf_tns = exe->get_gpu_buf(tensor);
    size_t tensor_size = tensor->shape_bytes();
    half_t *data = (half_t *)malloc(tensor_size);
    ark::gpu_memcpy(data, buf_tns, tensor_size);
    for (int i = 0; i < tensor->size(); ++i) {
        cout << data[i] << " ";
    }
    cout << endl;
    delete[] data;
}

class FullyConnectedLayer
{
  public:
    FullyConnectedLayer(int dim_input, int dim_output, TensorType dtype,
                        Model &model)
        : model{model}
    {
        Tensor *weight = model.tensor({dim_input, dim_output}, dtype);
        Tensor *bias = model.tensor({1, dim_output}, dtype);
        params = {weight, bias};
    }

    Tensor *forward(Tensor *input)
    {
        this->input = input;
        Tensor *weight = params[0];
        Tensor *output1 = model.matmul(input, weight);
        Tensor *bias = params[1];
        Tensor *output2 = model.add(output1, bias);
        return output2;
    }

    Tensor *backward(Tensor *grad)
    {

        Tensor *weight = params[0];
        Tensor *bias = params[1];
        Tensor *grad_output2 = grad;
        Tensor *grad_bias = model.tensor(bias->shape, bias->type);
        grad_bias = model.scale(grad_output2, 1, grad_bias);
        Tensor *grad_output1 = grad_output2;
        Tensor *grad_input = model.tensor(input->shape, input->type);
        Tensor *grad_weight = model.tensor(weight->shape, weight->type);
        grad_input =
            model.matmul(grad_output1, weight, nullptr, 1, false, true);
        grad_weight =
            model.matmul(input, grad_output1, nullptr, 1, true, false);
        grads[weight] = grad_weight;
        grads[bias] = grad_bias;
        return grad_input;
    }

    void apply_grads()
    {
        for (auto &param : params) {
            Tensor *grad = grads[param];
            // the learning rate
            model.add(param, model.scale(grad, -0.0001), param);
        }
    }

    void print_tensors(Executor *exe)
    {
        LOG(DEBUG, "input: ", input->name);
        print_tensor(input, exe);
        // print the parameters.
        for (int i = 0; i < params.size(); ++i) {
            LOG(DEBUG, "param: ", params[i]->name);
            print_tensor(params[i], exe);
        }
    }

    Tensor *input;
    vector<Tensor *> params;
    map<Tensor *, Tensor *> grads;
    Model &model;
};

class FFN_Model
{
  public:
    //
    FFN_Model(int dim_model, TensorType dtype, Model &model, int layer_num)
        : model{model}
    {
        for (int i = 0; i < layer_num; ++i) {
            FullyConnectedLayer layer{dim_model, dim_model, dtype, model};
            layers.push_back(layer);
        }
    }

    Model &get_model()
    {
        return model;
    }

    //
    Tensor *forward(Tensor *input = nullptr)
    {
        for (int i = 0; i < layers.size(); ++i) {
            LOG(DEBUG, "forward layer: ", i);
            input = layers[i].forward(input);
        }
        return input;
    }

    //
    void backward(Tensor *grad)
    {
        for (int i = layers.size() - 1; i >= 0; --i) {
            LOG(DEBUG, "backward layer: ", i);
            grad = layers[i].backward(grad);
        }
    }

    void print_tensors(Executor *exe)
    {
        for (int i = 0; i < layers.size(); ++i) {
            LOG(DEBUG, "layer: ", i);
            layers[i].print_tensors(exe);
        }
    }

    //
    Model &model;
    // model parameters.
    vector<FullyConnectedLayer> layers;
    //
    Tensor *model_input;

    //
};

class LossFn
{
  public:
    LossFn(Model &model) : model{model}
    {
    }

    Tensor *forward(Tensor *output, Tensor *ground_truth)
    {
        this->output = output;
        LOG(DEBUG, "loss forward");
        neg_ground_truth =
            model.tensor(ground_truth->shape, ground_truth->type);
        neg_ground_truth = model.scale(ground_truth, -1, neg_ground_truth);
        diff = model.tensor(output->shape, output->type);
        model.add(output, neg_ground_truth, diff);
        diff1 = model.tensor(diff->shape, diff->type);
        model.scale(diff, 1, diff1);
        LOG(DEBUG, "diff: ", diff->shape);
        loss_tensor = model.tensor(diff->shape, diff->type);
        model.mul(diff, diff1, loss_tensor);
        return loss_tensor;
    }

    Tensor *backward(Tensor *loss_tensor)
    {
        LOG(DEBUG, "loss backward");
        grad_diff = model.tensor(diff->shape, diff->type);
        model.mul(loss_tensor, diff, grad_diff);
        return grad_diff;
    }

    void print_tensors(Executor *exe)
    {
        LOG(DEBUG, "loss_fn.output: ");
        print_tensor(this->output, exe);
        LOG(DEBUG, "loss_fn.neg_ground_truth: ");
        print_tensor(this->neg_ground_truth, exe);
        LOG(DEBUG, "loss_fn.diff: ");
        print_tensor(this->diff, exe);
        LOG(DEBUG, "loss_fn.diff1: ");
        print_tensor(this->diff1, exe);
        LOG(DEBUG, "loss_fn.neg_ground_truth: ");
        print_tensor(this->neg_ground_truth, exe);
        LOG(DEBUG, "loss_fn.loss_tensor: ");
        print_tensor(this->loss_tensor, exe);
        LOG(DEBUG, "loss_fn.grad_diff: ");
        print_tensor(this->grad_diff, exe);
    }
    Tensor *output;
    Tensor *loss_tensor;
    Tensor *neg_ground_truth;
    Tensor *diff;
    Tensor *diff1;
    Tensor *grad_diff;
    Model &model;
};

class Trainer
{
  public:
    Trainer(Model &model, int dim_input, int batch_size, int gpu_id,
            int num_gpus)
        : ffn_model{dim_input, FP16, model, 2}, model{model}, loss_fn{model},
          batch_size{batch_size}, num_gpus{num_gpus}, gpu_id{gpu_id}
    {
        input = model.tensor({batch_size, dim_input}, FP16);
        ground_truth = model.tensor({batch_size, dim_input}, FP16);
        output = ffn_model.forward(input);
        loss_tensor = loss_fn.forward(output, ground_truth);
        grad_loss = model.tensor(loss_tensor->shape, loss_tensor->type);
        grad_output = loss_fn.backward(grad_loss);
        ffn_model.backward(grad_output);
        apply_grad();

        exe = new Executor(gpu_id, gpu_id, (int)num_gpus, model,
                           "sampleFFN_Model");
        LOG(DEBUG, "compile");
        exe->compile();
    }

    void init_data()
    {
        // init the input and ground_truth.
        ark::srand();
        auto data_input = range_halfs(this->input->shape_bytes(), 1, 0);
        exe->tensor_memcpy(this->input, data_input.get(),
                           this->input->shape_bytes());
        // LOG(DEBUG, "input: ");
        // print_tensor(this->input, this->exe);
        auto data_ground_truth =
            range_halfs(this->ground_truth->shape_bytes(), 2, 0);
        exe->tensor_memcpy(this->ground_truth, data_ground_truth.get(),
                           this->ground_truth->shape_bytes());
        // LOG(DEBUG, "ground_truth: ");
        // print_tensor(this->ground_truth, this->exe);
        // init the grad_loss with 1.
        auto data_grad_loss = range_halfs(this->grad_loss->shape_bytes(), 1, 0);
        exe->tensor_memcpy(this->grad_loss, data_grad_loss.get(),
                           this->grad_loss->shape_bytes());
        // LOG(DEBUG, "grad_loss: ");
        // print_tensor(this->grad_loss, this->exe);
        // init all the parameters of the model with random values.
        for (auto &layer : ffn_model.layers) {
            for (auto &param : layer.params) {
                auto data = rand_halfs(param->shape_bytes(), 1);
                exe->tensor_memcpy(param, data.get(), param->shape_bytes());
            }
        }
    }

    void train(int iter, int print_interval = 1)
    {
        exe->launch();
        if (print_interval == 0) {
            // don't print the loss for debug.
            exe->run(iter);
        } else {
            // we only print the loss every print_interval iterations for debug.
            for (int i = 0; i < iter; ++i) {
                exe->run(1);
                if (i % print_interval == 0) {
                    float loss = get_loss();
                    cout << "iter: " << i << ", loss: " << loss << endl;
                }
            }
        }
        float elapsed_msec = exe->stop();
        cout << "Elapsed: " << elapsed_msec / iter << " ms/iter\n";
    }

    float get_loss()
    {
        ark::GpuBuf *buf_tns = exe->get_gpu_buf(this->loss_tensor);
        size_t tensor_size = this->loss_tensor->shape_bytes();
        half_t *loss = (half_t *)malloc(tensor_size);

        ark::gpu_memcpy(loss, buf_tns, tensor_size);
        float loss_sum = 0;
        for (int i = 0; i < this->loss_tensor->size(); ++i) {
            loss_sum += (float)loss[i];
        }
        delete[] loss;
        return loss_sum;
    }

    void apply_grad()
    {
        for (auto &layer : ffn_model.layers) {
            layer.apply_grads();
        }
    }

    void print_tensors(Executor *exe)
    {
        LOG(DEBUG, "loss_tensor: ");
        print_tensor(this->loss_tensor, exe);
        LOG(DEBUG, "input: ");
        print_tensor(this->input, exe);
        LOG(DEBUG, "output: ");
        print_tensor(this->output, exe);
        LOG(DEBUG, "ground_truth: ");
        print_tensor(this->ground_truth, exe);
        LOG(DEBUG, "ffn_model: ");
        this->ffn_model.print_tensors(exe);
        LOG(DEBUG, "loss_fn: ");
        this->loss_fn.print_tensors(exe);
    }

    Tensor *loss_tensor, *input, *ground_truth, *output;
    Tensor *grad_output;
    Tensor *grad_loss;
    FFN_Model ffn_model;
    LossFn loss_fn;
    Model &model;
    Executor *exe;
    int batch_size;
    int num_gpus;
    int gpu_id;
};

int main(int argc, const char **argv)
{
    int batch_size = 1;
    int num_gpus = 1;
    int dims = 64;
    vector<int> pids;
    for (int gpu_id = 0; gpu_id < (int)num_gpus; ++gpu_id) {
        pids.emplace_back(proc_spawn([&] {
            Model model;
            Trainer trainer{model, dims, batch_size, gpu_id, num_gpus};
            trainer.init_data();
            LOG(DEBUG, "start training ");
            // train the model.
            trainer.train(10, 1);
            // trainer.print_tensors(trainer.exe);
            return 0;
        }));
    }
    int state = 0;
    for (auto pid : pids) {
        int ret = proc_wait(pid);
        if (ret != 0) {
            cerr << "E: Process " << pid << " returned " << ret << endl;
            state = 1;
        }
    }
    return state;
}