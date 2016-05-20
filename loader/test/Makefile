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

SRCS            := main.cpp thread_test.cpp datagen.cpp decode_test.cpp
TOBJS 			= $(subst .cpp,.o,$(SRCS))
OBJS 			= $(subst .c,.o,$(TOBJS))
CFLAGS          := -Wall -Wno-deprecated-declarations -O3
CXXFLAGS        := -Wall -Wno-deprecated-declarations -O3 -std=c++11
CFLAGSDBG       := -g -Wall -O0 -std=c++11
CC              := g++
UNAME_S         := $(shell uname -s)
DEPDIR = .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d
GTEST_TEST      := /usr/local/lib/libgtest.a
ifeq ($(UNAME_S), FreeBSD)
	CC          := clang++
endif

ifneq ("$(wildcard $(GTEST_TEST))","")
	GTEST_INSTALLED = true
else
	GTEST_INSTALLED = false
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

# hackery to fix bug in opencv. It exports gtest symbols in the opencv_ts library
IMGLIBS = $(subst -lopencv_ts,,$(shell pkg-config --libs-only-l opencv))

MEDIAFLAGS      := $(IMGFLAG) $(VIDFLAG) $(AUDFLAG)
LDIR            := $(IMGLDIR) $(VIDLDIR)
LIBS            := $(IMGLIBS) $(VIDLIBS) -lgtest -lpthread
INC             := $(INC) -I../src

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

$(shell mkdir -p bin)

.PHONY: clean

print-%  : ; @echo $* = $($*)

ifeq ($(GTEST_INSTALLED),true)
test: $(OBJS)
	@echo "Building $@..."
	$(CC) -o test $(OBJS) ../src/event.o ../src/media.o ../src/device.o ../src/batchfile.o ../src/image.o $(LDIR) $(LIBS)
else
test:
	@echo "gtest must be installed to build test"
endif

%.o : %.c $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CFLAGS) $(INC) $(DEPFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $<
	$(POSTCOMPILE)

%.o : %.cpp $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CXXFLAGS) $(INC) $(DEPFLAGS) $(GPUFLAG) $(MEDIAFLAGS) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))

clean:
	@rm -vf *.o
	@rm test
	@rm -rf $(DEPDIR)
