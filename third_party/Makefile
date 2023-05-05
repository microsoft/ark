# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

MKDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

GPUDMA_DIR := $(MKDIR)/gpudma
GPUDMA_NVIDIA_SRC ?= $(shell ls -d1 /usr/src/nvidia-*.*.* | tail -n 1)
GPUDMA_PATCHES       := $(shell find $(MKDIR)/patches/gpudma -name '*.patch')
GPUDMA_PATCH_TARGETS := $(patsubst $(MKDIR)/patches/gpudma/%.patch,$(MKDIR)/gpudma/%,$(GPUDMA_PATCHES))
GPUDMA_TARGET := $(GPUDMA_DIR)/module/gpumem.ko

TMPDIR := $(MKDIR)/.tmp


.PHONY: clean

gpudma: $(GPUDMA_TARGET)

$(GPUDMA_TARGET): $(GPUDMA_PATCHES) $(GPUDMA_PATCH_TARGETS)
	mkdir -p $(TMPDIR)/nvidia
	cp -r $(shell ls -d1 /usr/src/nvidia-*.*.* | tail -n 1) $(TMPDIR)/nvidia/kernel
	$(MAKE) -j -C $(TMPDIR)/nvidia/kernel
	for patchfile in $(GPUDMA_PATCHES); do \
		suf=$${patchfile#$(MKDIR)/patches/gpudma/} && patch -sN $(MKDIR)/gpudma/$${suf%.patch} -i $$patchfile -r /dev/null &> /dev/null; \
	done
	cd $(GPUDMA_DIR)/module && GPUDMA_DIR=$(TMPDIR) $(MAKE) -j
	rm -r $(TMPDIR)

clean:
	@$(MAKE) -C $(GPUDMA_DIR)/module -f $(GPUDMA_DIR)/module/Makefile clean
	@rm -rf $(TMPDIR)
