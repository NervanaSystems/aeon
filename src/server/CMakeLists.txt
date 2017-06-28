cmake_minimum_required(VERSION 2.8.11)
project(aeon-server)

set(SRC
    main.cpp
    manifest_builder.cpp
    gen_image.cpp
    aeonsvc.cpp
    status.cpp
    utils.cpp
    pidfile.cpp)

include_directories("${CMAKE_SOURCE_DIR}/src" "/usr/include/cpprest")
#include_directories("${CMAKE_SOURCE_DIR}/src" "/home/tpatejko/installs/casablanca/lib.debug/include")

find_package(OpenSSL 1.0.0 REQUIRED)

add_library(cpprest SHARED IMPORTED)
set_property(TARGET cpprest PROPERTY IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libcpprest.so")
 #set_property(TARGET cpprest PROPERTY IMPORTED_LOCATION "/home/tpatejko/installs/casablanca/lib.debug/lib/libcpprest.so")

add_executable(aeon-server ${SRC})
add_dependencies(aeon-server aeon)
target_link_libraries(aeon-server
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
