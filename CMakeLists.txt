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

cmake_minimum_required(VERSION 3.5)

project(aeon)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel"
        FORCE)
endif()

set(AEON_VERSION_MAJOR 1)
set(AEON_VERSION_MINOR 3)
set(AEON_VERSION_PATCH 1)

set(AEON_BUILD_NUMBER 0)

set(AEON_VERSION ${AEON_VERSION_MAJOR}.${AEON_VERSION_MINOR}.${AEON_VERSION_PATCH}
                 CACHE STRING "AEON logical version")

add_definitions(-DBUILD_VERSION_MAJOR=${AEON_VERSION_MAJOR}
                -DBUILD_VERSION_MINOR=${AEON_VERSION_MINOR}
                -DBUILD_VERSION_PATCH=${AEON_VERSION_PATCH}
                -DBUILD_BUILD_NUMBER=${AEON_BUILD_NUMBER})

option(ENABLE_AEON_SERVICE "Enable compilation of AEON Service" OFF)
option(ENABLE_AEON_CLIENT "Enable compilation of AEON Client" ON)
option(ENABLE_OPENFABRICS_CONNECTOR "Enable OpenFabrics connector for batch exchange between service and client" OFF)

set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
include(cmake/summary.cmake)
include(cmake/external_gtest.cmake)
include(cmake/external_json.cmake)

if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif()

if(NOT ${POSTFIX_VERSION} STREQUAL "")
    set(AEON_VERSION "${AEON_VERSION}${POSTFIX_VERSION}")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

option(COVERAGE "make code coverage" OFF)
if(COVERAGE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Debug")
    message(WARNING "Changing CMAKE_BUILD_TYPE to Debug because COVERAGE is set to ON")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(LCOVERAGE "--coverage")
endif(COVERAGE)

option(PYTHON_PLUGIN "Enable support of custom python plugins in augmentation pipeline" OFF)
if(PYTHON_PLUGIN)
    add_definitions(-DPYTHON_PLUGIN)
    list(APPEND CMAKE_CXX_FLAGS "-DPYTHON_PLUGIN")
endif(PYTHON_PLUGIN)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=inconsistent-missing-override")

# whitelist errors here
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

find_package(Threads)
find_package(PkgConfig REQUIRED)

pkg_check_modules(SOX REQUIRED sox)
include_directories(${SOX_INCLUDE_DIRS})
link_directories(${SOX_LIBRARY_DIRS})
add_definitions(${SOX_CFLAGS_OTHERS})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# opecv_ts is for testing opencv. if it is left in the lib list then we get an error when
# the unit test exits.
# Centos like os:
list(REMOVE_ITEM OpenCV_LIBRARIES "${OpenCV_LIB_DIR_OPT}/libopencv_ts.so")
# Ubuntu like os:
list(REMOVE_ITEM OpenCV_LIBRARIES opencv_ts)

find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})
link_directories(${CURL_LIBRARY_DIRS})

set(Python_ADDITIONAL_VERSIONS 3.6 3.5 3.4)

find_package(PythonLibs)
find_package(PythonInterp)

if (PYTHON_PLUGIN)
    if (NOT ${PYTHON_VERISON_MAJOR} EQUAL ${OpenCV_VERSION_MAJOR})
        message(WARNING "Python version " ${PYTHON_VERSION_MAJOR} " incompatible with OpenCV version " ${OpenCV_VERSION_MAJOR})
    endif ()
endif (PYTHON_PLUGIN)


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

if (ENABLE_AEON_SERVICE)
    # Check for C++ REST SDK from Microsoft
    find_package(CppRest REQUIRED)
    include_directories(${CPPREST_INCLUDE_DIRS})
    link_directories(${CPPREST_LIBRARY_DIRS})

    # Check for OpenSSL
    find_package(OpenSSL REQUIRED)
    include_directories(${OPENSSL_INCLUDE_DIRS})
    link_directories(${OPENSSL_LIBRARY_DIRS})
endif()

if (NOT ENABLE_AEON_SERVICE AND NOT ENABLE_AEON_CLIENT)
    set(ENABLE_OPENFABRICS_CONNECTOR OFF)
endif()

if (ENABLE_OPENFABRICS_CONNECTOR)
    # Check for OpenFabrics Interface
    find_package(OpenFabrics REQUIRED)
    include_directories(${OPENFABRICS_INCLUDE_DIRS})
    link_directories(${OPENFABRICS_LIBRARY_DIRS})
endif()

# Check for BOOST components
find_package(Boost REQUIRED COMPONENTS system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/test_data)

if (ENABLE_AEON_SERVICE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_AEON_SERVICE")
    if (ENABLE_OPENFABRICS_CONNECTOR)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_OPENFABRICS_CONNECTOR")
    endif()
endif()

if (ENABLE_AEON_CLIENT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_AEON_CLIENT")
endif()

# Check for documentation prerequisites
find_package(LATEX)
find_package(Doxygen)
find_package(Sphinx)
find_package(Breathe)

add_subdirectory(src)
if (ENABLE_AEON_SERVICE)
add_subdirectory(src/service)
endif()
include_directories(src)
if (ENABLE_AEON_CLIENT)
include_directories(src/client)
endif()
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)
include(cmake/clang-format.cmake)
include(cmake/coverage.cmake)

print_configuration_summary()
