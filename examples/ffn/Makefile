# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ARK_ROOT ?= /root/.ark
CUDIR    ?= /usr/local/cuda

CXX      := g++
CXXFLAGS := -std=c++14 -Wall -Wextra
INCLUDE  := -I$(ARK_ROOT)/include
LDFLAGS  := -L$(CUDIR)/lib64/stubs -Wl,-rpath,$(CUDIR)/lib64
LDLIBS   := -lcuda -lnvidia-ml -lnvrtc -lpthread -lrt -libverbs -lnuma

all: ffn

ffn: ffn.o
	$(CXX) -o $@ $< -L$(ARK_ROOT)/lib -lark $(LDFLAGS) $(LDLIBS)

ffn.o: ffn.cc
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDE) -c $<

clean:
	rm -f ffn ffn.o
