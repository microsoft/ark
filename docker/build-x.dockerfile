ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="ARK"
LABEL org.opencontainers.image.source https://github.com/microsoft/ark

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_VERSION="3.26.4"

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

# Install cmake 3.26.4
ENV CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64" \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
RUN curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local && \
    rm -rf ${CMAKE_HOME}.tar.gz
ENV PATH="/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"

ARG EXTRA_LD_PATH
ENV LD_LIBRARY_PATH="${EXTRA_LD_PATH}:${LD_LIBRARY_PATH}"

RUN echo PATH="${PATH}" > /etc/environment \
    echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment

ENTRYPOINT []
