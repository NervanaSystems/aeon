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
    main.cpp \
    gen_image.cpp \
    gen_audio.cpp \
    gen_video.cpp \
    test_bbox.cpp \
    test_image.cpp \
    test_image_var.cpp \
    test_label_map.cpp \
    test_pixel_mask.cpp \
    test_char_map.cpp \
    test_provider.cpp \
		test_provider_audio.cpp \
    test_util.cpp \
    test_video.cpp \
    test_types.cpp \
    test_params.cpp \
    manifest_test.cpp \
    manifest_maker.cpp \
    batch_file_loader_test.cpp \
    batch_loader_cpio_cache_test.cpp \
    buffer_test.cpp \
    shuffled_batch_iterator_test.cpp \
    helpers.cpp \
    simple_loader.cpp \
    mock_batch_loader.cpp \
    minibatch_iterator_test.cpp \
    test_localization.cpp \
    test_audio.cpp \
    test_logging.cpp \

OBJS             = $(subst .cpp,.o,$(TEST_SRCS))
INC             := -I../src $(INC)
# hackery to fix bug in opencv. It exports gtest symbols in the
# opencv_ts library so remove it from LIBS
LIBS            := $(subst -lopencv_ts,,$(LIBS))
LIBS            := $(LIBS) -lgtest -lpthread
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
