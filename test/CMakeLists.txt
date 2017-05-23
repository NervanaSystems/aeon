# gtest warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shift-sign-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-noreturn")

set (SRC
    gen_image.cpp
    helpers.cpp
    main.cpp
    manifest_builder.cpp
    test_async_manager.cpp
    test_audio.cpp
    test_augmentation.cpp
    test_base64.cpp
    test_bbox.cpp
    test_block_loader_file.cpp
    test_block_loader_nds.cpp
    test_block_manager.cpp
    test_buffer.cpp
    test_char_map.cpp
    test_config.cpp
    test_cpio.cpp
    test_file_util.cpp
    test_image.cpp
    test_label_map.cpp
    test_loader.cpp
    test_localization.cpp
    test_logging.cpp
    test_manifest_csv.cpp
    test_params.cpp
    test_pixel_mask.cpp
    test_provider_audio.cpp
    test_provider.cpp
    test_specgram.cpp
    test_types.cpp
    test_util.cpp
    test_video.cpp
    test_raw_image.cpp
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

add_executable(test ${SRC})
target_link_libraries(test aeon pthread sox ${CURL_LIBRARIES} gtest ${LCOVERAGE} ${OpenCV_LIBRARIES} ${PYTHON_LIBRARIES})
add_dependencies(test aeon)

add_custom_target(runtest
	COMMAND ${PROJECT_BINARY_DIR}/test/test
	DEPENDS test)
