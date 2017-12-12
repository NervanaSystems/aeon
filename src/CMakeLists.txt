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

set (SRC
    async_manager.cpp
    augment_audio.cpp
    augment_image.cpp
    avi.cpp
    base64.cpp
    batch_decoder.cpp
    batch_iterator.cpp
    block.cpp
    block_loader_file.cpp
    block_loader_nds.cpp
    block_manager.cpp
    box.cpp
    boundingbox.cpp
    buffer_batch.cpp
    cache_system.cpp
    cap_mjpeg_decoder.cpp
    cpio.cpp
    crc.cpp
    etl_audio.cpp
    etl_boundingbox.cpp
    etl_char_map.cpp
    etl_depthmap.cpp
    etl_image.cpp
    etl_label_map.cpp
    etl_localization_rcnn.cpp
    etl_localization_ssd.cpp
    etl_pixel_mask.cpp
    etl_video.cpp
    file_util.cpp
    image.cpp
    interface.cpp
    loader.cpp
    log.cpp
    manifest_file.cpp
    manifest_nds.cpp
    noise_clips.cpp
    normalized_box.cpp
    provider.cpp
    provider_factory.cpp
    specgram.cpp
    typemap.cpp
    util.cpp
    wav_data.cpp
    web_app.cpp
    web_server.cpp
    event.cpp
    raw_image.cpp)

if (ENABLE_AEON_CLIENT)
    list(APPEND SRC
        client/curl_connector.cpp
        client/loader_remote.cpp
        client/service.cpp
        client/remote_config.cpp)
endif()

if (ENABLE_OPENFABRICS_CONNECTOR)
    list(APPEND SRC
         rdma/ofi.cpp
         client/ofi_connector.cpp)
endif()

set(AEON_HEADER_FILES
        version.hpp)

if (PYTHON_FOUND)
    list(APPEND SRC api.cpp)
endif(PYTHON_FOUND)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/../setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_library(aeon SHARED ${SRC} ${AEON_HEADER_FILES})

install(TARGETS aeon DESTINATION lib)

set(AEON_LIBRARIES ${SOX_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBRARIES} ${PYTHON_LIBRARIES})
if (ENABLE_OPENFABRICS_CONNECTOR)
    list(APPEND AEON_LIBRARIES ${OPENFABRICS_LIBRARIES})
endif()

target_link_libraries(aeon ${AEON_LIBRARIES})
set_target_properties(aeon PROPERTIES VERSION ${AEON_VERSION_MAJOR}.${AEON_VERSION_MINOR}.${AEON_VERSION_PATCH}
                                      SOVERSION ${AEON_VERSION_MAJOR})
