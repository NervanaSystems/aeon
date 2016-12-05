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

print-%  : ; @echo $* = $($*)

HAS_GPU := $(shell nvcc --version > /dev/null 2>&1 && echo true)

bin/loader.so: Makefile
	@cd src && make ../bin/loader.so HAS_GPU=$(HAS_GPU) -j8

test: build_test
	@test/test $(ARGS)

build_test: Makefile
	@cd src && make loader.a HAS_GPU=$(HAS_GPU) -j8
	@cd test && make test HAS_GPU=$(HAS_GPU) -j8

install_test:
	@pip install flask

.PHONY: all test bin/loader.so build_test install_test

clean:
	@cd src  && make clean
	@cd test && make clean