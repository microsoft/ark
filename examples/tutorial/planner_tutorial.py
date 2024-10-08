# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import time
import torch
import torch.nn.functional as F


class VanillaSoftmax(ark.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        max = ark.reduce_max(input, axis=-1)
        output = ark.sub(input, max)
        output = ark.exp(output)
        sum = ark.reduce_sum(output, axis=-1)
        output = ark.div(output, sum)
        return output


class Softmax(ark.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        with ark.PlannerContext(
            warp_range=[0, 8],
            sram_range=[0, 0],
            sync=False,
            config={
                "NumWarps": 1,
                "SramBytes": 0,
                "NumTasks": 65536,
            },
        ):
            with ark.PlannerContext(config={"ImplType": "WarpWise"}):
                max = ark.reduce_max(input, axis=-1)
            with ark.PlannerContext(config={"Tile": [1, 2048]}):
                output = ark.sub(input, max)
                output = ark.exp(output)
            with ark.PlannerContext(config={"ImplType": "WarpWise"}):
                sum = ark.reduce_sum(output, axis=-1)
            with ark.PlannerContext(config={"Tile": [1, 2048]}):
                output = ark.div(output, sum)
            return output


def eval(tensor: ark.Tensor):
    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        return tensor.to_torch()


def perf(num_iter: int = 1000):
    with ark.Runtime() as rt:
        rt.launch()

        start = time.time()
        rt.run(iter=num_iter)
        end = time.time()
        return (end - start) / num_iter


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
