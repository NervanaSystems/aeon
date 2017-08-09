set(SRC
    iterator.cpp
)

file(COPY ${TEST_DATA_DIR}/flowers.jpg DESTINATION /tmp/)
file(COPY ${TEST_DATA_DIR}/img_2112_70.jpg DESTINATION /tmp/)

add_executable(iterator ${SRC})
target_link_libraries(iterator aeon pthread sox ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${PYTHON_LIBRARIES})
add_dependencies(iterator aeon)
