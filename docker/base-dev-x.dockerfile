ARG BASE_IMAGE=ghcr.io/microsoft/ark/ark:base-cuda12.1
FROM ${BASE_IMAGE}

LABEL maintainer="ARK"
LABEL org.opencontainers.image.source https://github.com/microsoft/ark

ENV ARK_SRC_DIR="/tmp/ark" \
    CMAKE_VERSION="3.26.4"

ADD . ${ARK_SRC_DIR}
WORKDIR ${ARK_SRC_DIR}

# Install Lcov
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lcov \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install cmake 3.26.4
ENV CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64" \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
RUN curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local && \
    rm -rf ${CMAKE_HOME}.tar.gz
ENV PATH="/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"

# Set PATH
RUN echo PATH="${PATH}" > /etc/environment

# Cleanup
WORKDIR /
RUN rm -rf ${ARK_SRC_DIR}
