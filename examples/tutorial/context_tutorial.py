# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import time
import torch
import torch.nn.functional as F


class VanillaSoftmax(ark.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, input):
        max = ark.reduce_max(input, axis=-1)
        output = ark.sub(input, max)
        output = ark.exp(output)
        sum = ark.reduce_sum(output, axis=-1)
        output = ark.div(output, sum)
        return output


class Softmax(ark.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, input):
        with ark.ContextManager(
            processor_range=[0, 304],
            warp_range=[0, 8],
            sram_range=[0, 0],
            task_id=0,
        ):
            max = ark.reduce_max(
                input,
                axis=-1,
                config={
                    "NumWarps": 1,
                    "ImplType": "WarpWise",
                    "SramBytes": 0,
                    "NumTasks": 65536,
                },
            )
            output = ark.sub(
                input,
                max,
                config={
                    "NumWarps": 1,
                    "SramBytes": 0,
                    "Tile": [1, 2048],
                    "NumTasks": 65536,
                },
            )
            output = ark.exp(
                output,
                config={
                    "NumWarps": 1,
                    "SramBytes": 0,
                    "Tile": [1, 2048],
                    "NumTasks": 65536,
                },
            )
            sum = ark.reduce_sum(
                output,
                axis=-1,
                config={
                    "NumWarps": 1,
                    "ImplType": "WarpWise",
                    "SramBytes": 0,
                    "NumTasks": 65536,
                },
            )
            output = ark.div(
                output,
                sum,
                config={
                    "NumWarps": 1,
                    "SramBytes": 0,
                    "Tile": [1, 2048],
                    "NumTasks": 65536,
                },
            )
            return output


def eval(tensor: ark.Tensor):
    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        return tensor.to_torch()


def perf():
    with ark.Runtime() as rt:
        rt.launch()

        start = time.time()
        rt.run(iter=1000)
        end = time.time()
        return (end - start) / 1000


if __name__ == "__main__":
    ark.init()

    shape = (32, 2048, 2048)

    input = torch.randn(*shape).to("cuda:0")

    output = Softmax()(ark.Tensor.from_torch(input))

    if torch.allclose(eval(output), F.softmax(input, dim=-1), atol=1e-5):
        print("Correct result")
    else:
        print("Incorrect result")

    print(f"Performance: {(perf() * 1e3):.3f} ms/iter")
