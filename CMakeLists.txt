# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

cmake_minimum_required(VERSION 2.8.12)

# Set this flag before project definition to avoid using other compiler by gtest
set(CMAKE_CXX_COMPILER "g++")

project(aeon C CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel"
        FORCE)
endif()

# Set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Get actual aeon version number from sources
include(AeonVersion)

AEON_GET_VERSION_LABEL()

string(REGEX MATCH "([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)(-(rc|dev)\\.[0-9?]+)?" AEON_VERSION "${AEON_VERSION_LABEL}")
string(REGEX REPLACE "-rc." "rc" AEON_VERSION "${AEON_VERSION}")
string(REGEX REPLACE "-dev." "dev" AEON_VERSION "${AEON_VERSION}")
string(REGEX MATCH "[^v](.*)" AEON_VERSION "${AEON_VERSION_LABEL}")
string(REPLACE "." ";" AEON_VERSION_PARTS "${AEON_VERSION}")
list(GET AEON_VERSION_PARTS 0 AEON_VERSION_MAJOR)
list(GET AEON_VERSION_PARTS 1 AEON_VERSION_MINOR)
list(GET AEON_VERSION_PARTS 2 AEON_VERSION_PATCH)

add_definitions(-DBUILD_VERSION_MAJOR=${AEON_VERSION_MAJOR}
                -DBUILD_VERSION_MINOR=${AEON_VERSION_MINOR}
                -DBUILD_VERSION_PATCH=${AEON_VERSION_PATCH})

option(SANITIZER_TYPE "Choose the type of sanitizer, options are: Address, Leak, Thread, Undefined" OFF)

include(cmake/summary.cmake)

if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

option(COVERAGE "make code coverage" OFF)
if(COVERAGE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Debug")
    message(WARNING "Changing CMAKE_BUILD_TYPE to Debug because COVERAGE is set to ON")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(LCOVERAGE "--coverage")
endif(COVERAGE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-z,now")

if (SANITIZER_TYPE AND NOT "${SANITIZER_TYPE}" MATCHES "^(Address|Leak|Thread|Undefined)$")
  message("Choose the correct type of sanitizer")
  return()
endif()

# Only one sanitizer is allowed in compile time
if(SANITIZER_TYPE)
    string(TOLOWER "${SANITIZER_TYPE}" sanitizer_type)
    if(sanitizer_type STREQUAL "address")
        set(fsanitize "-fsanitize=address")
    elseif(sanitizer_type STREQUAL "leak")
        set(fsanitize "-fsanitize=leak")
    elseif(sanitizer_type STREQUAL "thread")
        set(fsanitize "-fsanitize=thread")
    elseif(sanitizer_type STREQUAL "undefined")
        set(fsanitize "-fsanitize=undefined")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${fsanitize}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=inconsistent-missing-override")

# whitelist errors here
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-disabled-macro-expansion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-old-style-cast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation-unknown-command")

# should remove these
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-member-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-enum") # from numpy
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-compat-deprecated-writable-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-double-promotion")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreserved-id-macro")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

find_package(Threads)
find_package(PkgConfig REQUIRED)

find_package(OpenCV REQUIRED)

if(NOT (${OpenCV_VERSION} LESS 3))
    add_definitions(-DOPENCV_LEGACY_MODE)
endif(GREATER_EQUAL)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

message("${OpenCV_INCLUDE_DIRS}")

# opecv_ts is for testing opencv. if it is left in the lib list then we get an error when
# the unit test exits.
# Centos like os:
list(REMOVE_ITEM OpenCV_LIBRARIES "${OpenCV_LIB_DIR_OPT}/libopencv_ts.so")
# Ubuntu like os:
list(REMOVE_ITEM OpenCV_LIBRARIES opencv_ts)

# set(Python_ADDITIONAL_VERSIONS 3.6 3.5 3.4)

find_package(PythonInterp)
find_package(PythonLibs)

if (PYTHONLIBS_FOUND)
    find_package(NumPy)
    if(NUMPY_FOUND)
        set(PYTHON_FOUND true)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON_FOUND")
        include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
        link_directories(${PYTHON_LIBRARY_DIRS})
        include_directories(SYSTEM ${NUMPY_INCLUDE_DIRS})
        link_directories(${NUMPY_LIBRARY_DIRS})
    endif(NUMPY_FOUND)
else()
    unset(PYTHON_LIBRARIES)
endif (PYTHONLIBS_FOUND)

# Check for BOOST components
find_package(Boost REQUIRED COMPONENTS system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/test_data)

# Check for documentation prerequisites
find_package(LATEX)
find_package(Doxygen)
find_package(Sphinx)
find_package(Breathe)

include_directories(BEFORE gtest/googletest/include)
include_directories(BEFORE gtest/googlemock/include)
set(BUILD_SHARED_LIBS off)
add_subdirectory(gtest)
set(BUILD_SHARED_LIBS on)
add_subdirectory(src)
include_directories(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)
include(cmake/clang-format.cmake)
include(cmake/coverage.cmake)

print_configuration_summary()
