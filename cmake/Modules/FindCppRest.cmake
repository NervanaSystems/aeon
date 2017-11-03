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

unset(CPPREST_VERSION)
unset(CPPREST_LIBRARY_DIRS)
unset(CPPREST_INCLUDE_DIRS)
unset(CPPREST_LIBRARIES)

find_path(CPPREST_INCLUDE_DIRS
    NAMES cpprest/http_client.h
    PATHS ${CPPREST_DIR} $ENV{CPPREST_DIR} /usr/include /usr/local/include
    PATH_SUFFIXES Release/include include)

find_library(_CPPREST_LIBRARY_DIRS
    NAMES libcpprest.so
    PATHS ${CPPREST_DIR} $ENV{CPPREST_DIR} /usr/lib /usr/local/lib
    PATH_SUFFIXES lib64 x86_64-linux-gnu lib build.release/Binaries)

if (_CPPREST_LIBRARY_DIRS AND CPPREST_INCLUDE_DIRS)
    find_file(CPPREST_VERSION_FILE cpprest/version.h ${CPPREST_INCLUDE_DIRS})
    if (CPPREST_VERSION_FILE)
        file(READ "${CPPREST_VERSION_FILE}" _CPPREST_VERSION_FILE_CONTENTS)
        string(REGEX MATCH "#define CPPREST_VERSION_MAJOR ([0-9])" _MATCH "${_CPPREST_VERSION_FILE_CONTENTS}")
        set(__cpprest_major ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define CPPREST_VERSION_MINOR ([0-9])" _MATCH "${_CPPREST_VERSION_FILE_CONTENTS}")
        set(__cpprest_minor ${CMAKE_MATCH_1})
        set(CPPREST_VERSION ${__cpprest_major}.${__cpprest_minor})
        if (CPPREST_VERSION AND CPPREST_VERSION VERSION_LESS CPPREST_FIND_VERSION)
            unset(CPPREST_VERSION)
            unset(CPPREST_INCLUDE_DIRS)
            unset(CPPREST_LIBRARY_DIRS)
            unset(CPPREST_LIBRARIES)
            message(WARNING "CppREST: installed version ${CPPREST_VERSION} does not meet the minimum required version of ${CPPREST_FIND_VERSION}")
        else()
            set(CPPREST_LIBRARIES ${_CPPREST_LIBRARY_DIRS})
            get_filename_component(CPPREST_LIBRARY_DIRS ${_CPPREST_LIBRARY_DIRS} DIRECTORY)
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CppREST REQUIRED_VARS CPPREST_LIBRARIES CPPREST_INCLUDE_DIRS CPPREST_LIBRARY_DIRS CPPREST_VERSION
                                          VERSION_VAR CPPREST_VERSION)

