set(SRC
    cpp_iterator.cpp
)

file(COPY ${TEST_DATA_DIR}/flowers.jpg DESTINATION .)
file(COPY ${TEST_DATA_DIR}/img_2112_70.jpg DESTINATION .)

add_executable(plugin_iterator ${SRC})
target_link_libraries(plugin_iterator aeon pthread ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${Python_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(plugin_iterator aeon)
