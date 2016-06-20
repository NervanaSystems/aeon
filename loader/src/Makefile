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

SRCS            := \
    media.cpp \
    event.cpp \
    device.cpp \
    batchfile.cpp \
    image.cpp \
    buffer.cpp \
    loader.cpp \
    argtype.cpp \
    etl_image.cpp \
    etl_bbox.cpp \
    etl_lmap.cpp \
    etl_specgram.cpp \
    etl_rawmedia.cpp \
    etl_video.cpp \
    codec.cpp \
    video.cpp \
    util.cpp \
    manifest.cpp \
    batch_file_loader.cpp \
    batch_loader_cpio_cache.cpp \
    sequential_batch_iterator.cpp \
    shuffled_batch_iterator.cpp \
    batch_loader.cpp \
    vector.cpp \

OBJS             = $(subst .cpp,.o,$(SRCS))
CFLAGS          := -Wall -Wno-deprecated-declarations -O3 -std=c++11 -fPIC
CFLAGSDBG       := -g -Wall -O0 -std=c++11
CC              := clang++
LOADER_SO       := ../bin/loader.so
LOADER_STATIC   := loader.a
UNAME_S         := $(shell uname -s)
DEPDIR             = .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d
ifeq ($(UNAME_S), FreeBSD)
	CC          := clang++
endif

ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
	IMGFLAG     := -DHAS_IMGLIB
	INC         := $(shell pkg-config --cflags opencv)
	IMGLDIR     := $(shell pkg-config --libs-only-L opencv)
	IMGLIBS     := $(shell pkg-config --libs-only-l opencv)
ifeq ($(shell pkg-config --exists libavutil libavformat libavcodec libswscale; echo $$?), 0)
	VIDFLAG     := -DHAS_VIDLIB
	AUDFLAG     := -DHAS_AUDLIB
	VIDLDIR     := $(shell pkg-config --libs-only-L libavutil libavformat libavcodec libswscale)
	VIDLIBS     := -lavutil -lavformat -lavcodec -lswscale
endif
endif

MEDIAFLAGS      := $(IMGFLAG) $(VIDFLAG) $(AUDFLAG)
LDIR            := $(IMGLDIR) $(VIDLDIR)
LIBS            := $(IMGLIBS) $(VIDLIBS)

ifeq ($(HAS_GPU), true)
ifeq ($(CUDA_ROOT),)
	CUDA_ROOT   := $(patsubst %/bin/nvcc,%, $(shell which nvcc))
endif

	GPUFLAG     := -DHAS_GPU
	INC         := -I$(CUDA_ROOT)/include $(INC)
	ifeq ($(shell uname -s),Darwin)
		LDIR    := -L$(CUDA_ROOT)/lib $(LDIR)
	else
		LDIR    := -L$(CUDA_ROOT)/lib64 $(LDIR)
	endif
	LIBS        := -lcuda -lcudart $(LIBS)
endif

.PHONY: clean

# g++ -o bin/loader -DSTANDALONE -g -Wall -O0 -std=c++11  -DHAS_IMGLIB -DHAS_VIDLIB  src/loader.cpp -I/usr/include/opencv       -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab   -lavutil -lavformat -lavcodec -lswscale

all: ../bin/loader.so $(LOADER_SO) $(LOADER_STATIC) Makefile

print-%  : ; @echo $* = $($*)

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
