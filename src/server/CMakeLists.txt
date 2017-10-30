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

# Check for C++ REST API (Casablanca)
pkg_check_modules(CPPREST cpprest)

if (NOT CPPREST_FOUND)
    set(CPPREST_LIBRARIES cpprest)
    find_path(CPPREST_INCLUDE_DIRS
          NAMES cpprest/http_client.h
          PATHS ${CPPREST_DIR} $ENV{CPPREST_DIR} ${CASABLANKA_DIR} $ENV{CASABLANKA_DIR} /usr/local/include /usr/include
          PATH_SUFFIXES Release/include include)
    find_library(CPPREST_LIBRARY_DIRS
          NAME libcpprest.so
          PATHS ${CPPREST_DIR} $ENV{CPPREST_DIR} ${CASABLANKA_DIR} $ENV{CASABLABKA_DIR} /usr/local /usr
          PATH_SUFFIXES lib Release/build.release/Binaries build.release/Binaries)
endif(NOT CPPREST_FOUND)

include_directories(${CPPREST_INCLUDE_DIRS})
link_directories(${CPPREST_LIBRARY_DIRS})

# Check for BOOST components
find_package(Boost REQUIRED COMPONENTS system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# Check for OpenSSL
find_package(OpenSSL REQUIRED)

include_directories(${OPENSSL_INCLUDE_DIRS})
link_directories(${OPENSSL_LIBRARY_DIRS})

set(AEON_SERVICE_SOURCES daemon.cpp server.cpp)
set(AEON_SERVICE_HEADERS dataset.hpp server.hpp)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(aeon-service ${AEON_SERVICE_SOURCES} ${AEON_SERVICE_HEADERS})
target_link_libraries(aeon-service aeon pthread ${Boost_LIBRARIES} ${CPPREST_LIBRARIES} ${OPENSSL_LIBRARIES})
