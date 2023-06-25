# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ARK_ROOT ?= /root/.ark
CUDIR    ?= /usr/local/cuda

CXX      := g++
CXXFLAGS := -std=c++14 -Wall -Wextra
INCLUDE  := -I$(ARK_ROOT)/include -I $(CUDIR)/include -I$(ARK_ROOT)/include/kernels
LDFLAGS  := -L$(CUDIR)/lib64/stubs -Wl,-rpath,$(CUDIR)/lib64
LDLIBS   := -lcuda -lnvidia-ml -lnvrtc -lpthread -lrt -libverbs -lnuma

all: build/tutorial
	
build/tutorial: build/tutorial.o
	$(CXX) -o $@ $< -L$(ARK_ROOT)/lib -lark $(LDFLAGS) $(LDLIBS)

build/tutorial.o: tutorial.cc
	mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDE) -c $<

clean:
	rm -r build/
