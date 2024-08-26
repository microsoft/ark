# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from ark.torch.tracer import tracer as ark_torch_tracer


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


@ark_torch_tracer
class ForwardPass(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ff = FeedForward(dim, hidden_dim)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        t = self.ff(input)
        return nn.functional.mse_loss(t, target)


def main():
    batch_size = 128
    num_batches = 1
    dim = 1024
    hidden_dim = 4096
    num_epochs = 10
    torch.manual_seed(42)
    torch.set_default_device("cuda:0")

    model = ForwardPass(dim=dim, hidden_dim=hidden_dim)

    inputs = [torch.randn(batch_size, dim) for _ in range(num_batches)]
    targets = [torch.randn(batch_size, dim) for _ in range(num_batches)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        optimizer.zero_grad()
        avg_loss = 0
        for input, target in zip(inputs, targets):
            loss = model(input, target)
            avg_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        avg_loss /= num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        avg_loss = 0
        for input, target in zip(inputs, targets):
            loss = model.forward_ark(input, target)
            avg_loss += loss.to_numpy()[0]
            model.backward_ark(loss)
            optimizer.step()
        avg_loss /= num_batches
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
