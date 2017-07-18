cmake_minimum_required (VERSION 2.8.11)

# Set this flag before project definition to avoid using other compiler by gtest
set(CMAKE_CXX_COMPILER "clang++")

project (aeon)

set(AEON_VERSION 1.2.0)
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
  if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    set(CMAKE_BUILD_TYPE "DEBUG")
    message(WARNING "Changing CMAKE_BUILD_TYPE to DEBUG because COVERAGE is set to ON")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(LCOVERAGE "--coverage")
endif(COVERAGE)

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=inconsistent-missing-override")

# whitelist errors here
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-disabled-macro-expansion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-old-style-cast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation-unknown-command")

# should remove these
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicate-enum") # from numpy
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-used-but-marked-unused") # from sox
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-compat-deprecated-writable-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-double-promotion")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

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
 
if (PYTHONLIBS_FOUND)
    find_package(NumPy)
    if(NUMPY_FOUND)
        set(PYTHON_FOUND true)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON_FOUND")
        include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
        link_directories(${PYTHON_LIBRARY_DIRS})
        include_directories(SYSTEM ${NUMPY_INCLUDE_DIRS})
        link_directories(${NUMPY_LIBRARY_DIRS})
    else()
        message("Numpy not found, Python interface not included")
    endif(NUMPY_FOUND)
else()
    message("Python not found, Python interface not included")
    set(PYTHON_LIBRARIES "")
endif (PYTHONLIBS_FOUND)

set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/test_data)

include_directories(BEFORE gtest/googletest/include/)
include_directories(BEFORE gtest/googlemock/include)
set(BUILD_SHARED_LIBS off)
add_subdirectory(gtest)
set(BUILD_SHARED_LIBS on)
add_subdirectory(src)
add_subdirectory(src/server)
add_subdirectory(src/client)
include_directories(src)
include_directories(src/client)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)
include(clang-format.cmake)
include(coverage.cmake)
