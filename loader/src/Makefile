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

OBJS             = $(subst .cpp,.o,$(SRCS))
LOADER_SO       := ../bin/loader.so
LOADER_STATIC   := loader.a

all: ../bin/loader.so $(LOADER_SO) $(LOADER_STATIC) Makefile

%.o : %.cpp $(DEPDIR)/%.d
	$(CC) -c -o $@ $(CFLAGS) $(INC) $(DEPFLAGS) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))

$(LOADER_SO): $(OBJS)
	@echo "Building $@..."
	@mkdir -p ../bin
	$(CC) -shared -o $@ $(OBJS) $(LDIR) ${DEPFLAGS} $(LIBS)

$(LOADER_STATIC): $(OBJS)
	@echo "Building $@..."
	ar rcs $@ $(OBJS)

clean:
	@rm -vf *.o $(LOADER_SO) $(LOADER_STATIC)