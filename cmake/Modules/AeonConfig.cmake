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

unset(Aeon_VERSION)
unset(Aeon_LIBRARY_DIRS)
unset(Aeon_INCLUDE_DIRS)
unset(Aeon_LIBRARIES)

find_path(Aeon_INCLUDE_DIRS
    NAMES aeon.hpp
    PATHS ${Aeon_DIR} $ENV{Aeon_DIR} /usr/include /usr/local/include
    PATH_SUFFIXES aeon build.release/src build.debug/src)

find_library(_Aeon_LIBRARY_DIRS
    NAMES libaeon.so
    PATHS ${Aeon_DIR} $ENV{Aeon_DIR} /usr/lib /usr/local/lib
    PATH_SUFFIXES lib64 x86_64-linux-gnu build.release/src build.debug/src)

if (_Aeon_LIBRARY_DIRS AND Aeon_INCLUDE_DIRS)
    file(READ "${Aeon_INCLUDE_DIRS}/aeon.hpp" _Aeon_VERSION_FILE_CONTENTS)
    string(REGEX MATCH "#define VERSION_MAJOR ([0-9])+" _MATCH "${_Aeon_VERSION_FILE_CONTENTS}")
    set(Aeon_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define VERSION_MINOR ([0-9])+" _MATCH "${_Aeon_VERSION_FILE_CONTENTS}")
    set(Aeon_VERSION_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define VERSION_PATCH ([0-9])+" _MATCH "${_Aeon_VERSION_FILE_CONTENTS}")
    set(Aeon_VERSION_PATCH ${CMAKE_MATCH_1})
    set(Aeon_VERSION ${Aeon_VERSION_MAJOR}.${Aeon_VERSION_MINOR}.${Aeon_VERSION_PATCH})
    if (Aeon_VERSION AND Aeon_VERSION VERSION_LESS Aeon_FIND_VERSION)
        unset(Aeon_VERSION)
        unset(Aeon_INCLUDE_DIRS)
        unset(Aeon_LIBRARY_DIRS)
        unset(Aeon_LIBRARIES)
        message(WARNING "Aeon: installed version ${Aeon_VERSION} does not meet the minimum required version of ${Aeon_FIND_VERSION}")
    else()
        set(Aeon_LIBRARIES ${_Aeon_LIBRARY_DIRS})
        get_filename_component(Aeon_LIBRARY_DIRS ${_Aeon_LIBRARY_DIRS} DIRECTORY)
    endif()
else()
    unset(Aeon_VESION)
    unset(Aeon_INCLUDE_DIRS)
    unset(Aeon_LIBRARY_DIRS)
    unset(Aeon_LIBRARIES)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Aeon REQUIRED_VARS Aeon_LIBRARIES Aeon_INCLUDE_DIRS Aeon_LIBRARY_DIRS Aeon_VERSION
                                       VERSION_VAR Aeon_VERSION)

mark_as_advanced(Aeon_VERSION Aeon_INCLUDE_DIRS Aeon_LIBRARY_DIRS Aeon_LIBRARIES)

include(FeatureSummary)
set_package_properties(Aeon PROPERTIES
    URL "http://github.com/NervanaSystem/aeon"
    DESCRIPTION "Framework-independent data loader for machine learning.")

