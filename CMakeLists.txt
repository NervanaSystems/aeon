cmake_minimum_required (VERSION 2.8.11)
project (aeon)

set(AEON_VERSION 1.0.0)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")

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

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# opecv_ts is for testing opencv. if it is left in the lib list then we get an error when
# the unit test exits.
list(REMOVE_ITEM OpenCV_LIBRARIES opencv_ts)

find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

find_package(PythonLibs)
find_package(PythonInterp)
 
if (PYTHONLIBS_FOUND)
    find_package(NumPy)
    if(NUMPY_FOUND)
        set(PYTHON_FOUND true)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON_FOUND")
        include_directories(${PYTHON_INCLUDE_DIR})
        include_directories(${NUMPY_INCLUDE_DIRS})
    else()
        message("Numpy not found, Python interface not included")
    endif(NUMPY_FOUND)
else()
    message("Python not found, Python interface not included")
    set(PYTHON_LIBRARIES "")
endif (PYTHONLIBS_FOUND)

set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/test_data)

add_subdirectory(src)
include_directories(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)

