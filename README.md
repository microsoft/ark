# ARK

A GPU-driven system framework for scalable AI applications.

[![Latest Release](https://img.shields.io/github/release/microsoft/ark.svg)](https://github.com/microsoft/ark/releases/latest)
[![License](https://img.shields.io/github/license/microsoft/ark.svg)](LICENSE)
[![CodeQL](https://github.com/microsoft/ark/actions/workflows/codeql.yml/badge.svg)](https://github.com/microsoft/ark/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/microsoft/ark/graph/badge.svg?token=XmMOK85GOB)](https://codecov.io/gh/microsoft/ark)

| Pipelines         | Build Status      |
|-------------------|-------------------|
| Unit Tests (CUDA) | [![Build Status](https://dev.azure.com/binyli/HPC/_apis/build/status%2Fark-test?branchName=main)](https://dev.azure.com/binyli/HPC/_build/latest?definitionId=6&branchName=main) |
| Unit Tests (ROCm) | [![Unit Tests (ROCm)](https://github.com/microsoft/ark/actions/workflows/ut-rocm.yml/badge.svg?branch=main)](https://github.com/microsoft/ark/actions/workflows/ut-rocm.yml) |

*NOTE (Nov 2023): ROCm unit tests will be replaced into an Azure pipeline in the future.*

*NOTE (Dec 2023): ROCm unit tests are failing due to the nodes' issue. This will be fixed soon.*

See [Quick Start](docs/quickstart.md) to quickly get started.

## Overview

ARK is a deep learning framework especially designed for highly optimized performance over distributed GPUs. Specifically, ARK adopts a GPU-driven execution model, where the GPU autonomously schedule and execute both computation and communication without any CPU intervention.

ARK provides a set of APIs for users to express their distributed deep learning applications. ARK then automatically schedules a GPU-driven execution plan for the application, which generates a GPU kernel code called *loop kernel*. The loop kernel is a GPU kernel that contains a loop that iteratively executes the entire application, including both computation and communication. ARK then executes the loop kernel on the distributed GPUs.

<img src="./docs/imgs/GPU-driven_System_Architecture.svg" alt="GPU-driven System Architecture" style="width: 900px;"/>

## Status & Roadmap

ARK is under active development and a part of its features will be added in a future release. The following describes key features of each version.

### New in ARK v0.5 (Latest Release)

* Integrate with [MSCCL++](https://github.com/microsoft/mscclpp)
* Removed dependency on `gpudma`
* Add AMD CDNA3 architecture support
* Support communication for AMD GPUs
* Optimize OpGraph scheduling
* Add a multi-GPU Llama2 example

See details from https://github.com/microsoft/ark/issues/168.

### ARK v0.6 (TBU, Jan. 2024)

* Overall performance optimization
* Improve Python unit tests & code coverage

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citations

<img src="./docs/imgs/logos.svg" alt="KAIST and Microsoft Logos" style="width: 350px;"/>

ARK is a collaborative research initiative between KAIST and Microsoft Research.
If you use this project in your research, please cite our [NSDI'23 paper]:

```bibtex
@inproceedings{HwangPSQCX23,
  author    = {Changho Hwang and
               KyoungSoo Park and
               Ran Shu and
               Xinyuan Qu and
               Peng Cheng and
               Yongqiang Xiong},
  title     = {ARK: GPU-driven Code Execution for Distributed Deep Learning},
  booktitle = {20th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 23)},
  year      = {2023},
  publisher = {{USENIX} Association},
}
```

[NSDI'23 paper]: https://www.usenix.org/conference/nsdi23/presentation/hwang
