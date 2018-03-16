# ******************************************************************************
# Copyright 2017-1018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

unset(NLOHMANN_JSON_VERSION)
unset(NLOHMANN_JSON_INCLUDE_DIRS)

find_path(NLOHMANN_JSON_INCLUDE_DIRS
    NAMES json.hpp nlohmann/json.hpp
    PATHS ${NLOHMANN_JSON_DIR} $ENV{NLOHMANN_JSON_DIR} /usr/include /usr/local/include)

if (NLOHMANN_JSON_INCLUDE_DIRS)
    file(READ ${NLOHMANN_JSON_INCLUDE_DIRS}/json.hpp __nlohmann_json_file_contents)
    string(REGEX MATCH "  version ([0-9]+)\\.([0-9]+)\\.([0-9]+)" __nlohmann_json_version_check "${__nlohmann_json_file_contents}")
    set(NLOHMANN_JSON_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nlohmann::Json REQUIRED_VARS NLOHMANN_JSON_VERSION NLOHMANN_JSON_INCLUDE_DIRS
    VERSION_VAR NLOHMANN_JSON_VERSION)
