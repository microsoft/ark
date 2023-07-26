# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ARK_ROOT ?= /usr/local/ark
CUDIR    ?= /usr/local/cuda

CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
INCLUDE  := -I$(ARK_ROOT)/include -I $(CUDIR)/include -I$(ARK_ROOT)/include/kernels
LDFLAGS  := -L$(CUDIR)/lib64/stubs -Wl,-rpath,$(CUDIR)/lib64
LDLIBS   := -lcuda -lnvidia-ml -lnvrtc -lpthread -lrt -libverbs -lnuma

all: build/ffn
	
build/ffn: build/ffn.o
	$(CXX) -o $@ $< -L$(ARK_ROOT)/lib -lark $(LDFLAGS) $(LDLIBS)

build/ffn.o: ffn.cc
	mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDE) -c $<

clean:
	rm -r build/
