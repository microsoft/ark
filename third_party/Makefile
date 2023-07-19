# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

MKDIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

GPUDMA_DIR := $(MKDIR)/gpudma
GPUDMA_NVIDIA_SRC ?= $(shell ls -d1 /usr/src/nvidia-*.*.* | tail -n 1)
GPUDMA_PATCHES       := $(shell find $(MKDIR)/patches/gpudma -name '*.patch')
GPUDMA_PATCH_TARGETS := $(patsubst %,patch-%,$(GPUDMA_PATCHES))
GPUDMA_TARGET := $(GPUDMA_DIR)/module/gpumem.ko

CUTLASS_DIR := $(MKDIR)/cutlass
CUTLASS_PATCHES       := $(shell find $(MKDIR)/patches/cutlass -name '*.patch')
CUTLASS_PATCH_TARGETS := $(patsubst %,patch-%,$(CUTLASS_PATCHES))

TMPDIR := $(MKDIR)/.tmp


.PHONY: cutlass clean

gpudma: $(GPUDMA_TARGET)

$(GPUDMA_TARGET): $(GPUDMA_PATCHES)
	mkdir -p $(TMPDIR)/nvidia
	cp -r $(shell ls -d1 /usr/src/nvidia-*.*.* | tail -n 1) $(TMPDIR)/nvidia/kernel
	$(MAKE) -j -C $(TMPDIR)/nvidia/kernel
	$(MAKE) -j $(GPUDMA_PATCH_TARGETS)
	cd $(GPUDMA_DIR)/module && GPUDMA_DIR=$(TMPDIR) $(MAKE) -j
	rm -r $(TMPDIR)

cutlass:
	@$(MAKE) -j $(CUTLASS_PATCH_TARGETS)

patch-$(MKDIR)/patches/%.patch: $(MKDIR)/% $(MKDIR)/patches/%.patch
	@patch -sN $< -i $(word 2,$^) -r /dev/null | true

clean:
	$(MAKE) -C $(GPUDMA_DIR)/module -f $(GPUDMA_DIR)/module/Makefile clean
	rm -rf $(TMPDIR)
	cd $(GPUDMA_DIR) && git reset --hard
	cd $(CUTLASS_DIR) && git reset --hard
