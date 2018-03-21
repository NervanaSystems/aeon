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

if (APPLE AND POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()

project(plugin_iterator)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel"
        FORCE)
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.1")
    add_compiler_options(-std=c++11)
else()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(Aeon REQUIRED)
include_directories(${Aeon_INCLUDE_DIRS})
link_directories(${Aeon_LIBRARY_DIRS})

set(PLUGIN_ITERATOR_SOURCE_FILES plugin_iterator.cpp)

add_executable(plugin_iterator ${PLUGIN_ITERATOR_SOURCE_FILES})
target_link_libraries(plugin_iterator ${Aeon_LIBRARIES})

