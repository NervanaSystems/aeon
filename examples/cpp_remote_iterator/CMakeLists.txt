project(remote_iterator)

set(SRC
    remote_iterator.cpp
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../server/)

add_executable(remote_iterator ${SRC})

add_dependencies(remote_iterator aeon)
add_dependencies(remote_iterator aeon-server)

add_custom_command(TARGET remote_iterator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/src/server/aeon-server ${CMAKE_CURRENT_BINARY_DIR}/../server/)

target_link_libraries(remote_iterator aeon pthread sox ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${PYTHON_LIBRARIES})
