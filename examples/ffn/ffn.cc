// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
// #include "ark/logging.h"
// #include "ark/ops/ops_test_utils.h"
using namespace std;
using namespace ark;

// Spawn a process that runs `func`. Returns PID of the spawned process.
int proc_spawn(const function<int()> &func)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    } else if (pid == 0) {
        int ret = func();
        std::exit(ret);
    }
    return (int)pid;
}

// Wait for a spawned process with PID `pid`.
// Return -1 on any unexpected failure, otherwise return the exit status.
int proc_wait(int pid)
{
    int status;
    if (waitpid(pid, &status, 0) == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
}

// Wait for multiple child processes.
// Return 0 on success, -1 on any unexpected failure, otherwise the first seen
// non-zero exit status.
int proc_wait(const vector<int> &pids)
{
    int ret = 0;
    for (auto &pid : pids) {
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            return -1;
        }
        int r;
        if (WIFEXITED(status)) {
            r = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            r = -1;
        } else {
            r = -1;
        }
        if ((ret == 0) && (r != 0)) {
            ret = r;
        }
    }
    return ret;
}

void print_tensor(Tensor *tensor, Executor *exe)
{
    cout << "tensor: " << tensor->name << endl;
    size_t tensor_size = tensor->shape_bytes();
    half_t *data = (half_t *)malloc(tensor_size);
    exe->tensor_memcpy(data, tensor, tensor_size);
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
        print_tensor(input, exe);
        // print the parameters.
        for (size_t i = 0; i < params.size(); ++i) {
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
    FFN_Model(int dim_model, TensorType dtype, Model &model, int layer_num,
              int num_gpus, int gpu_id)
        : model{model}, num_gpus{num_gpus}, gpu_id{gpu_id}
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
        for (size_t i = 0; i < layers.size(); ++i) {
            printf("forward layer: %d\n", i);
            input = layers[i].forward(input);
        }
        return input;
    }

    //
    void backward(Tensor *grad)
    {
        for (int i = layers.size() - 1; i >= 0; --i) {
            printf("backward layer: %d\n", i);
            grad = layers[i].backward(grad);
        }
        DimType grads_size = 0;
        vector<Tensor *> grads;

        for (auto &layer : layers) {
            for (auto &param : layer.params) {
                grads.push_back(layer.grads[param]);
                grads_size += layer.grads[param]->size();
            }
        }

        // All-reduce gradients
        if (num_gpus > 1) {
            Tensor *gradients = model.tensor({1, grads_size, 1, 1}, FP16);
            Tensor *idn = model.identity(gradients, {grads});

            model.all_reduce(idn, gpu_id, num_gpus);
        }
    }

    void print_tensors(Executor *exe)
    {
        for (size_t i = 0; i < layers.size(); ++i) {
            printf("layer: %d\n", i);
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
    int num_gpus;
    //
    int gpu_id;
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
        printf("loss forward");
        neg_ground_truth =
            model.tensor(ground_truth->shape, ground_truth->type);
        neg_ground_truth = model.scale(ground_truth, -1, neg_ground_truth);
        diff = model.tensor(output->shape, output->type);
        model.add(output, neg_ground_truth, diff);
        diff1 = model.tensor(diff->shape, diff->type);
        model.scale(diff, 1, diff1);
        loss_tensor = model.tensor(diff->shape, diff->type);
        model.mul(diff, diff1, loss_tensor);
        return loss_tensor;
    }

    Tensor *backward(Tensor *loss_tensor)
    {
        printf("loss backward");
        grad_diff = model.tensor(diff->shape, diff->type);
        model.mul(loss_tensor, diff, grad_diff);
        return grad_diff;
    }

    void print_tensors(Executor *exe)
    {
        printf("loss_fn.output: ");
        print_tensor(this->output, exe);
        printf("loss_fn.neg_ground_truth: ");
        print_tensor(this->neg_ground_truth, exe);
        printf("loss_fn.diff: ");
        print_tensor(this->diff, exe);
        printf("loss_fn.diff1: ");
        print_tensor(this->diff1, exe);
        printf("loss_fn.neg_ground_truth: ");
        print_tensor(this->neg_ground_truth, exe);
        printf("loss_fn.loss_tensor: ");
        print_tensor(this->loss_tensor, exe);
        printf("loss_fn.grad_diff: ");
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
        : model{model}, ffn_model{dim_input, FP16, model, 2, num_gpus, gpu_id},
          loss_fn{model},
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
        exe->compile();
    }

    void init_data()
    {
        // init the input and ground_truth.
        auto data_input = range_halfs(this->input->shape_bytes(), 1, 0);
        exe->tensor_memcpy(this->input, data_input.get(),
                           this->input->shape_bytes());
        // printf( "input: ");
        // print_tensor(this->input, this->exe);
        auto data_ground_truth =
            range_halfs(this->ground_truth->shape_bytes(), 2, 0);
        exe->tensor_memcpy(this->ground_truth, data_ground_truth.get(),
                           this->ground_truth->shape_bytes());
        // printf( "ground_truth: ");
        // print_tensor(this->ground_truth, this->exe);
        // init the grad_loss with 1.
        auto data_grad_loss = range_halfs(this->grad_loss->shape_bytes(), 1, 0);
        exe->tensor_memcpy(this->grad_loss, data_grad_loss.get(),
                           this->grad_loss->shape_bytes());
        // printf( "grad_loss: ");
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
                exe->wait();
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
        size_t tensor_size = this->loss_tensor->shape_bytes();
        half_t *loss = (half_t *)malloc(tensor_size);
        exe->tensor_memcpy(loss, this->loss_tensor, tensor_size);
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
        printf("loss_tensor: ");
        print_tensor(this->loss_tensor, exe);
        printf("input: ");
        print_tensor(this->input, exe);
        printf("output: ");
        print_tensor(this->output, exe);
        printf("ground_truth: ");
        print_tensor(this->ground_truth, exe);
        printf("ffn_model: ");
        this->ffn_model.print_tensors(exe);
        printf("loss_fn: ");
        this->loss_fn.print_tensors(exe);
    }

    Model &model;
    Tensor *loss_tensor, *input, *ground_truth, *output;
    Tensor *grad_output;
    Tensor *grad_loss;
    FFN_Model ffn_model;
    LossFn loss_fn;
    Executor *exe;
    int batch_size;
    int num_gpus;
    int gpu_id;
};

struct Args
{
    int batch_size;
    int dims;
    int num_gpus;
    int iterations;
    int print_interval;
    int seed;
    bool verbose;
};

Args parse_args(int argc, const char **argv)
{
    string prog = argv[0];
    vector<string> args(argv + 1, argv + argc);

    auto print_help = [&prog]() {
        cerr << "Usage: " << prog << " [options]\n"
             << "Options:\n"
             << "  -h, --help\t\t\tPrint this help message\n"
             << "  -b, --batch-size <int>\t\tBatch size\n"
             << "  -d, --dims <int>\t\tDimensions\n"
             << "  -g, --num-gpus <int>\t\tNumber of GPUs\n"
             << "  -i, --iter <int>\t\tNumber of iterations\n"
             << "  -p, --print-interval <int>\tPrint interval\n"
             << "  -s, --seed <int>\t\tRandom seed\n"
             << "  -v, --verbose\t\t\tVerbose output\n";
        exit(0);
    };

    Args ret;

    // Default arguments
    ret.batch_size = 1;
    ret.dims = 64;
    ret.num_gpus = 1;
    ret.iterations = 10;
    ret.print_interval = 1;
    ret.seed = -1;
    ret.verbose = false;

    for (auto it = args.begin(); it != args.end(); ++it) {
        if (*it == "-h" || *it == "--help") {
            print_help();
        } else if (*it == "-b" || *it == "--batch-size") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.batch_size = stoi(*it);
        } else if (*it == "-d" || *it == "--dims") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.dims = stoi(*it);
        } else if (*it == "-g" || *it == "--num-gpus") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.num_gpus = stoi(*it);
        } else if (*it == "-i" || *it == "--iter") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.iterations = stoi(*it);
        } else if (*it == "-p" || *it == "--print-interval") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.print_interval = stoi(*it);
        } else if (*it == "-s" || *it == "--seed") {
            if (++it == args.end()) {
                cerr << "Error: missing argument for " << *(it - 1) << endl;
                exit(1);
            }
            ret.seed = stoi(*it);
        } else if (*it == "-v" || *it == "--verbose") {
            ret.verbose = true;
        } else {
            cerr << "Error: unknown option " << *it << endl;
            print_help();
        }
    }

    return ret;
}

int main(int argc, const char **argv)
{
    Args args = parse_args(argc, argv);

    cout << "--" << endl
         << "batch_size=" << args.batch_size << endl
         << "dims=" << args.dims << endl
         << "num_gpus=" << args.num_gpus << endl
         << "iterations=" << args.iterations << endl
         << "print_interval=" << args.print_interval << endl
         << "seed=" << args.seed << endl
         << "verbose=" << args.verbose << endl
         << "--" << endl;

    vector<int> pids;
    for (int gpu_id = 0; gpu_id < args.num_gpus; ++gpu_id) {
        pids.emplace_back(proc_spawn([&] {
            // ark::srand(args.seed);

            Model model;
            Trainer trainer{model, args.dims, args.batch_size, gpu_id,
                            args.num_gpus};
            trainer.init_data();
            // train the model.
            trainer.train(args.iterations, args.print_interval);
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
