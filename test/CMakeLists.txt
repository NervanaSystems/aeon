#
# Copyright 2017 Intel(R) Nervana(TM)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
    test_localization_rcnn.cpp
    test_localization_ssd.cpp
    test_logging.cpp
    test_manifest_tsv.cpp
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

set(TESTSUITE_LIBRARIES aeon ${SOX_LIBRARIES} gtest gmock
    ${LCOVERAGE} ${OpenCV_LIBRARIES} ${PYTHON_LIBRARIES}
    ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (ENABLE_AEON_SERVICE)
    list(APPEND SRC
        ../src/service/service.cpp
        test_curl_connector.cpp
        test_service.cpp
        test_loader_remote.cpp)

    list(APPEND TESTSUITE_LIBRARIES
        ${CURL_LIBRARIES} ${CPPREST_LIBRARIES} ${OPENSSL_LIBRARIES})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

add_executable(testsuite ${SRC})
target_link_libraries(testsuite ${TESTSUITE_LIBRARIES})
add_dependencies(testsuite aeon)

add_custom_target(runtest
	COMMAND ${PROJECT_BINARY_DIR}/test/testsuite
	DEPENDS testsuite)
