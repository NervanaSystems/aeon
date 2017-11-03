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

unset(NUMPY_VERSION)
unset(NUMPY_INCLUDE_DIRS)

if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
  endif()
endif()

if(PYTHONINTERP_FOUND)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                    "import numpy as n; print(n.__version__); print(n.get_include());"
                    RESULT_VARIABLE __numpy_result
                    OUTPUT_VARIABLE __numpy_output
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (__numpy_result MATCHES 0)
        string(REGEX REPLACE ";" "\\\\;" __numpy_values ${__numpy_output})
        string(REGEX REPLACE "\r?\n" ";" __numpy_values ${__numpy_values})
        list(GET __numpy_values 0 NUMPY_VERSION)
        list(GET __numpy_values 1 NUMPY_INCLUDE_DIRS)
        string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __numpy_version_check "${NUMPY_VERSION}")
        if (NOT "${__numpy_version_check}" STREQUAL "")
            set(NUMPY_VERSION_MAJOR ${CMAKE_MATCH_1})
            set(NUMPY_VERSION_MINOR ${CMAKE_MATCH_2})
            set(NUMPY_VERSION_PATCH ${CMAKE_MATCH_3})
        else()
            set(NUMPY_VERSION "?unknown?")
            message(WARNING "NumPy: problem extracting version!")
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy REQUIRED_VARS NUMPY_INCLUDE_DIRS NUMPY_VERSION
                                        VERSION_VAR NUMPY_VERSION)

