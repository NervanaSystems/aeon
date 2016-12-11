# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

include ../Makefile.base

.PHONY: clean

# specific to TEST

TEST_SRCS := \
    test_buffer.cpp \
    csv_manifest_maker.cpp \
    test_manifest_csv.cpp \
    gen_image.cpp \
    helpers.cpp \
    main.cpp \
    test_audio.cpp \
    test_bbox.cpp \
    test_block_loader_async.cpp \
    test_block_manager.cpp \
    test_char_map.cpp \
    test_image.cpp \
    test_label_map.cpp \
    test_localization.cpp \
    test_logging.cpp \
    test_params.cpp \
    test_pixel_mask.cpp \
    test_provider.cpp \
    test_provider_audio.cpp \
    test_types.cpp \
    test_util.cpp \
    test_video.cpp \
    test_config.cpp \
    test_cpio.cpp \
    test_file_util.cpp \
    test_loader.cpp \
    test_async_manager.cpp \
    # test_cpio_cache.cpp \
    # test_batch_iterator.cpp \
    # test_block_iterator_shuffled.cpp \
    # test_block_loader_cpio_cache.cpp \
    # test_block_loader_file.cpp \
    # test_block_loader_nds.cpp \
    # block_loader_util.cpp \

OBJS             = $(subst .cpp,.o,$(TEST_SRCS))
INC             := -I../src $(INC)
# hackery to fix bug in opencv. It exports gtest symbols in the
# opencv_ts library so remove it from LIBS
LIBS            := $(subst -lopencv_ts,,$(LIBS))
LIBS            := $(LIBS) -lsox -lgtest -lpthread
GTEST_TEST      := /usr/local/lib/libgtest.a
LOADER_LIB      := ../src/loader.a
CFLAGS          := $(CFLAGS) -DCURDIR=\"$(CURDIR)\"

ifneq ("$(wildcard $(GTEST_TEST))","")
	GTEST_INSTALLED = true
else
	GTEST_INSTALLED = false
endif

ifeq ($(GTEST_INSTALLED),true)
test: $(OBJS) $(LOADER_LIB)
	@echo "Building $@..."
	$(CC) -o test $(OBJS) $(LOADER_LIB) $(LDIR) $(LIBS)
else
test:
	@echo "gtest must be installed to build test"
endif

%.o : %.cpp $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CFLAGS) $(INC) $(DEPFLAGS) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TEST_SRCS)))

clean:
	@rm -vf *.o
	@rm -f test
	@rm -rf $(DEPDIR)
	@rm -rf audio_data
	@rm -rf video_data
	@rm -rf image_data