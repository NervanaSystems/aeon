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

unset(BREATHE_EXECUTABLE)
unset(BREATHE_VERSION)

find_program(BREATHE_EXECUTABLE NAMES breathe-apidoc DOC "Path to breathe executable")

if (BREATHE_EXECUTABLE)
    execute_process(COMMAND "${BREATHE_EXECUTABLE}" "--version"
                    RESULT_VARIABLE __breathe_result OUTPUT_VARIABLE __breathe_output ERROR_VARIABLE __breathe_error
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    if (__breathe_result MATCHES 0)
        if ("${__breathe_output}" STREQUAL "")
            set(__breathe_output "${__breathe_error}")
        endif()
        string(REPLACE " " ";" __breathe_values "${__breathe_output}")
        list(GET __breathe_values 2 BREATHE_VERSION)
    else()
        string(REGEX MATCH "DistributionNotFound: (.+)" __breathe_match "${__breathe_error}")
        if (NOT "${__breathe_match}" STREQUAL "")
            set(BREATHE_MISSING_REQUIREMENTS "${CMAKE_MATCH_1}")
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Breathe REQUIRED_VARS BREATHE_EXECUTABLE BREATHE_VERSION
                                          VERSION_VAR BREATHE_VERSION)
