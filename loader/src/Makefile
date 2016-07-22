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

# specific to src

SRCS := \
    cpio.cpp \
    buffer_in.cpp \
    buffer_out.cpp \
    pyBackendWrapper.cpp \
    pyLoader.cpp \
    api_pversion.cpp \
    etl_audio.cpp \
    noise_clips.cpp \
    specgram.cpp \
    log.cpp \
    codec.cpp \
    image.cpp \
    etl_pixel_mask.cpp \
    etl_image.cpp \
    etl_image_var.cpp \
    etl_bbox.cpp \
    etl_label_map.cpp \
    etl_char_map.cpp \
    etl_localization.cpp \
    etl_video.cpp \
    util.cpp \
    wav_data.cpp \
    manifest.cpp \
    provider_factory.cpp \
    batch_file_loader.cpp \
    batch_loader_cpio_cache.cpp \
    sequential_batch_iterator.cpp \
    shuffled_batch_iterator.cpp \
    minibatch_iterator.cpp \
    batch_loader.cpp \
    box.cpp \
    buffer_pool.cpp \
    buffer_pool_in.cpp \
    buffer_pool_out.cpp \
    cap_mjpeg_decoder.cpp \

OBJS             = $(subst .cpp,.o,$(SRCS))
LOADER_SO       := ../bin/loader.so
LOADER_STATIC   := loader.a

all: ../bin/loader.so $(LOADER_SO) $(LOADER_STATIC) Makefile

%.o : %.c $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CFLAGS) $(INC) $(DEPFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $<
	$(POSTCOMPILE)

%.o : %.cpp $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CFLAGS) $(INC) $(DEPFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))

$(LOADER_SO): $(OBJS)
	@echo "Building $@..."
	@mkdir -p ../bin
	$(CC) -shared -o $@ $(OBJS) $(LDIR) $(LIBS)

$(LOADER_STATIC): $(OBJS)
	@echo "Building $@..."
	ar rcs $@ $(OBJS)

clean:
	@rm -vf *.o $(LOADER_SO) $(LOADER_STATIC)
