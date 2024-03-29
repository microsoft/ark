# Temporal Dockerfile for building ARK base image for ROCm 5.6

ARG BASE_IMAGE=rocm/dev-ubuntu-20.04:5.6.1-complete
FROM ${BASE_IMAGE}

LABEL maintainer="ARK"
LABEL org.opencontainers.image.source https://github.com/microsoft/ark

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /opt/nvidia

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcap2 \
        libnuma-dev \
        openssh-client \
        openssh-server \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        sudo \
        wget \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install OFED
ENV OFED_VERSION=5.2-2.2.3.0
RUN cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install OpenMPI
ENV OPENMPI_VERSION=4.1.5
RUN cd /tmp && \
    export ompi_v_parsed="$(echo ${OPENMPI_VERSION} | sed -E 's/^([0-9]+)\.([0-9]+)\..*/\1.\2/')" && \
    wget -q https://download.open-mpi.org/release/open-mpi/v${ompi_v_parsed}/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/mpi && \
    make -j && \
    make install && \
    cd .. && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}*

ARG EXTRA_LD_PATH=/opt/rocm/lib
ENV PATH="/usr/local/mpi/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/mpi/lib:${EXTRA_LD_PATH}:${LD_LIBRARY_PATH}"

RUN echo PATH="${PATH}" > /etc/environment && \
    echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment

# Copy amd_hip_bf16.h from ROCm 5.7
ADD amd_hip_bf16.h /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h

ENTRYPOINT []
