cmake_minimum_required(VERSION 2.8.11)
project(aeon-server)

set(SRC_LIB
    server.cpp
    status.cpp
    utils.cpp
    pidfile.cpp)

include_directories("${CMAKE_SOURCE_DIR}/src" "/usr/include/cpprest")

find_package(OpenSSL REQUIRED)

add_library(cpprest SHARED IMPORTED)
set_property(TARGET cpprest PROPERTY IMPORTED_LOCATION "/usr/local/lib/libcpprest.so")

add_library(aeon-server-lib STATIC ${SRC_LIB})
target_link_libraries(aeon-server-lib
    aeon
    pthread
    sox
    opencv_imgproc
    opencv_highgui
    opencv_core
    cpprest
    boost_system
    ssl
    crypto
    ${PYTHON_LIBRARIES}
    ${CURL_LIBRARIES}
)

set(SRC
    aeonsvc.cpp)

add_executable(aeon-server ${SRC})
add_dependencies(aeon-server-lib aeon)
add_dependencies(aeon-server aeon-server-lib)
target_link_libraries(aeon-server aeon-server-lib)
