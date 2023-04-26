# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

MKDIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARKDIR  := $(MKDIR)
CUDIR   := /usr/local/cuda
MPIDIR  := /usr/local/mpi
KAHYPAR ?= 0

CXX      := g++
CXXFLAGS := -std=c++14 -Wall -g -O3 -fPIC
INCLUDE  := -I $(CUDIR)/include -I $(ARKDIR) -I $(ARKDIR)/third_party
INCLUDE  += -I $(ARKDIR)/third_party/cutlass/include
INCLUDE  += -I $(MPIDIR)/include
LDLIBS   := -lcuda -lnvidia-ml -lnvrtc -lpthread -lrt -libverbs -lnuma
LDFLAGS  := -L $(CUDIR)/lib64/stubs -Wl,-rpath,$(CUDIR)/lib64
MACROS   :=

BDIR ?= $(ARKDIR)/build

BSRC_IPC := ipc_mem.cc ipc_lock.cc ipc_shm.cc ipc_coll.cc ipc_socket.cc ipc_hosts.cc
BSRC_NET := net_ib.cc
BSRC_GPU := gpu_mem.cc gpu_buf.cc gpu_comm_sw.cc gpu_mgr.cc
BSRC_GPU += gpu_kernel.cc gpu_compile.cc
BSRC_SCHED := sched_op.cc sched_opseq.cc sched_codegen.cc sched_opgraph.cc sched_profiler.cc sched_tile.cc
BSRC_SCHED_SCHED := sched_default.cc sched_simple.cc
ifeq ($(KAHYPAR),1)
BSRC_SCHED_SCHED += sched_kahypar.cc
endif
BSRC_UNITTEST := unittest_utils.cc

BSRC_OPS := ops_common.cc ops_config.cc ops_test_utils.cc ops_tensor.cc ops_identity.cc ops_reshape.cc
BSRC_OPS += ops_add.cc ops_mul.cc ops_scale.cc ops_reduce.cc ops_matmul.cc ops_linear.cc ops_im2col.cc
BSRC_OPS += ops_conv.cc ops_max_pool.cc ops_sendrecv.cc ops_all_reduce.cc ops_sendrecv_mm.cc ops_transpose.cc

BSRC := init.cc cpu_timer.cc logging.cc math.cc random.cc env.cc file_io.cc process.cc
BSRC += model.cc model_io.cc tensor.cc dims.cc
BSRC += executor.cc

BSRC += $(patsubst %.cc,ipc/%.cc,$(BSRC_IPC))
BSRC += $(patsubst %.cc,net/%.cc,$(BSRC_NET))
BSRC += $(patsubst %.cc,gpu/%.cc,$(BSRC_GPU))
BSRC += $(patsubst %.cc,ops/%.cc,$(BSRC_OPS))
BSRC += $(patsubst %.cc,unittest/%.cc,$(BSRC_UNITTEST))
BSRC += $(patsubst %.cc,sched/%.cc,$(BSRC_SCHED))
BSRC += $(patsubst %.cc,sched/sched/%.cc,$(BSRC_SCHED_SCHED))

BOBJ := $(patsubst %.cc,$(BDIR)/ark/%.o,$(BSRC))

USRC_OPS := ops_tensor_test.cc ops_identity_test.cc ops_reshape_test.cc ops_add_test.cc ops_mul_test.cc ops_reduce_test.cc ops_all_reduce_test.cc ops_scale_test.cc
USRC_OPS += ops_im2col_test.cc ops_matmul_test.cc ops_dot_test.cc ops_sendrecv_mm_test.cc ops_transpose_test.cc

USRC := ipc/ipc_mem_test.cc ipc/ipc_coll_test.cc ipc/ipc_socket_test.cc
USRC += gpu/gpu_mem_test.cc gpu/gpu_mgr_test.cc gpu/gpu_kernel_test.cc
USRC += net/net_ib_test.cc dims_test.cc sched/sched_test.cc

USRC += $(patsubst %.cc,ops/%.cc,$(USRC_OPS))

UOBJ := $(patsubst %.cc,$(BDIR)/ark/%.o,$(USRC))
UBIN := $(patsubst %.o,%,$(UOBJ))

# SSRC := all_reduce.cc transformer_dp.cc transformer_pp.cc gpu_comm.cc
# SSRC += bert.cc resnet50.cc googlenet.cc ssd.cc
SSRC := ffn.cc
SOBJ := $(patsubst %.cc,$(BDIR)/samples/%.o,$(SSRC))
SBIN := $(patsubst %.o,%,$(SOBJ))

ifeq ($(KAHYPAR),1)
KHP_BDIR := $(BDIR)/third_party/kahypar
KHP_SO := $(KHP_BDIR)/kahypar/build/install/lib/libkahypar.so
KHP_SO += $(KHP_BDIR)/boost_1_69_0/stage/lib/libboost_program_options.so.1.69.0
MACROS += -DUSE_KAHYPAR
else
KHP_SO :=
endif

CPPSOURCES := $(shell find $(ARKDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu)' -not -path "*/build/*" -not -path "*/third_party/*" -not -path "*/tests/*")

LIBNAME   := libark.so
LIBTARGET := $(BDIR)/lib/$(LIBNAME)

.PHONY: all build third_party submodules kahypar cutlass gpudma samples unittest clean

all: build unittest

third_party: cutlass | submodules

build: $(BOBJ) lib
unittest: $(UBIN)
samples: $(SBIN)

submodules:
	@git submodule update --init --recursive

kahypar: | submodules
	@ARKDIR=$(ARKDIR) BDIR=$(BDIR) ./scripts/build_kahypar.sh

cutlass: | submodules

gpudma: | submodules
	@ARKDIR=$(ARKDIR) ./scripts/build_gpudma.sh

cpplint:
	clang-format-12 -style=file --verbose --Werror --dry-run $(CPPSOURCES)

cpplint-autofix:
	clang-format-12 -style=file --verbose --Werror -i $(CPPSOURCES)

$(UBIN): %: %.o $(BOBJ) | third_party
	$(CXX) -o $@ $(LDFLAGS) $< $(BOBJ) $(KHP_SO) $(LDLIBS)

$(SBIN): %: %.o $(LIBTARGET) | third_party
	$(CXX) -o $@ $(LDFLAGS) -L $(MPIDIR)/lib $< $(LIBTARGET) $(KHP_SO) $(LDLIBS) -lmpi

$(BDIR)/%.o: %.cc | third_party
	@mkdir -p $(@D)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDE) -c $< $(MACROS)

lib: $(BOBJ)
	@mkdir -p $(BDIR)/lib
	$(CXX) -shared -o $(LIBTARGET) $(BOBJ)

install:
	@ARKDIR=$(ARKDIR) ARK_ROOT=$(ARK_ROOT) ./scripts/install.sh

clean:
	rm -rf $(BDIR)/ark $(BDIR)/samples $(BDIR)/lib
