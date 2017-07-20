set(SRC
    remote_iterator.cpp
)

file(COPY ${TEST_DATA_DIR}/flowers.jpg DESTINATION .)
file(COPY ${TEST_DATA_DIR}/img_2112_70.jpg DESTINATION .)

add_executable(remote_iterator ${SRC})
target_link_libraries(remote_iterator aeon pthread sox ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${PYTHON_LIBRARIES})
add_dependencies(remote_iterator aeon)
