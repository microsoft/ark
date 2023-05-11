# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ARK_ROOT ?= /root/.ark
CUDIR    ?= /usr/local/cuda

CXX      := g++
CXXFLAGS := -std=c++14 -Wall -Wextra -g
INCLUDE  := -I$(ARK_ROOT)/include -I $(CUDIR)/include -I$(ARK_ROOT)/include/kernels
LDFLAGS  := -L$(CUDIR)/lib64/stubs -Wl,-rpath,$(CUDIR)/lib64
LDLIBS   := -lcuda -lnvidia-ml -lnvrtc -lpthread -lrt -libverbs -lnuma

all: test_interface

test_interface: test_interface.o
	$(CXX) -o $@ $< -L$(ARK_ROOT)/lib -lark $(LDFLAGS) $(LDLIBS)

test_interface.o: test_interface.cc
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDE) -c $<

clean:
	rm -f test_interface test_interface.o
