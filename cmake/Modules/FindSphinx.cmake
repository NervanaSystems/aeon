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

unset(SPHINX_EXECUTABLE)
unset(SPHINX_VERSION)

find_program(SPHINX_EXECUTABLE NAMES sphinx-build sphinx-build2 DOC "Path to sphinx-build executable")

if (SPHINX_EXECUTABLE)
    execute_process(COMMAND "${SPHINX_EXECUTABLE}"
                    RESULT_VARIABLE __sphinx_result OUTPUT_VARIABLE __sphinx_output ERROR_VARIABLE __sphinx_error
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    if ("${__sphinx_output}" STREQUAL "")
        set(__sphinx_output "${__sphinx_error}")
    endif()
    if (__sphinx_result MATCHES 1)
        string(REGEX MATCH "Sphinx v([0-9]+)\\.([0-9]+)\\.([0-9]+)" __sphinx_version_check "${__sphinx_output}")
        set(SPHINX_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
    else()
        unset(SPHINX_VERSION "?unknown?")
        message(WARNING "Sphinx: problem extracting version!")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx REQUIRED_VARS SPHINX_EXECUTABLE SPHINX_VERSION
                                  VERSION_VAR SPHINX_VERSION)

option(SPHINX_OUTPUT_HTML "Output standalone HTML files" ON)
option(SPHINX_OUTPUT_MAN "Output man pages" ON)
option(SPHINX_WARNINGS_AS_ERRORS "When building documentation treat warnings as errors" ON)
