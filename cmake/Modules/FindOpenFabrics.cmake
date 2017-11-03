#
# Copyright 2017 Intel(R) Nervana(TM)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

unset(OPENFABRICS_VERSION)
unset(OPENFABRICS_INCLUDE_DIRS)
unset(OPENFABRICS_LIBRARY_DIRS)
unset(OPENFABRICS_LIBRARIES)

pkg_check_modules(OPENFABRICS QUIET fabric)

if (NOT OPENFABRICS_FOUND)
    find_path(OPENFABRICS_INCLUDE_DIRS
              NAMES rdma/fi_endpoint.h
              PATHS ${OPENFABRICS_DIR} $ENV{OPENFABRICS_DIR} /usr/local/include /usr/include
              PATH_SUFFIXES build.release/include Release/include include)
    find_library(_OPENFABRICS_LIBRARY_DIRS
              NAMES libfabric.so
              PATHS ${OPENFABRICS_DIR} $ENV{OPENFABRICS_DIR} /usr/local /usr
              PATH_SUFFIXES lib64 lib/x86_64-linux-gnu lib build.release)
    if (_OPENFABRICS_LIBRARY_DIRS AND OPENFABRICS_INCLUDE_DIRS)
        find_file(OPENFABRICS_VERSION_FILE rdma/fabric.h ${OPENFABRICS_INCLUDE_DIRS})
        if (OPENFABRICS_VERSION_FILE)
            file(READ "${OPENFABRICS_VERSION_FILE}" _OPENFABRICS_VERSION_FILE_CONTENTS)
            string(REGEX MATCH "#define FI_MAJOR_VERSION ([0-9])" _MATCH "${_OPENFABRICS_VERSION_FILE_CONTENTS}")
            set(OPENFABRICS_VERSION_MAJOR ${CMAKE_MATCH_1})
            string(REGEX MATCH "#define FI_MINOR_VERSION ([0-9])" _MATCH "${_OPENFABRICS_VERSION_FILE_CONTENTS}")
            set(OPENFABRICS_VERSION_MINOR ${CMAKE_MATCH_1})
            set(OPENFABRICS_VERSION ${OPENFABRICS_VERSION_MAJOR}.${OPENFABRICS_VERSION_MINOR})
            if (OPENFABRICS_VERSION AND OPENFABRICS_VESION VERSION_LESS OPENFABRICS_FIND_VERSION)
                unset(OPENFABRICS_VERSION)
                unset(OPENFABRICS_INCLUDE_DIRS)
                unset(OPENFABRICS_LIBRARY_DIRS)
                unset(OPENFABRICS_LIBRARIES)
                message(STATUS "OpenFabrics: installed version ${OPENFABRICS_VERSION} does not meet the minimum required version of ${OPENFABRICS_FIND_VERSION}")
            else()
                set(OPENFABRICS_LIBRARIES ${_OPENFABRICS_LIBRARY_DIRS})
                get_filename_component(OPENFABRICS_LIBRARY_DIRS ${_OPENFABRICS_LIBRARY_DIRS} DIRECTORY)
            endif()
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenFabrics REQUIRED_VARS OPENFABRICS_LIBRARIES OPENFABRICS_INCLUDE_DIRS OPENFABRICS_LIBRARY_DIRS OPENFABRICS_VERSION
                                              VERSION_VAR OPENFABRICS_VERSION)

