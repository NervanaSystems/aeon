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
    raw_image.cpp
)

file(GLOB DEPLOY_HEADERS_ABS *.hpp)
foreach(DEPLOY_HEADER ${DEPLOY_HEADERS_ABS})
    get_filename_component(FNAME ${DEPLOY_HEADER} NAME)
    set(DEPLOY_HEADERS ${DEPLOY_HEADERS} ${FNAME})
endforeach()

if (PYTHON_FOUND)
set(SRC ${SRC} api.cpp)
endif(PYTHON_FOUND)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/../setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_library(aeon SHARED ${SRC})

install(TARGETS aeon DESTINATION lib)
install(FILES ${DEPLOY_HEADERS} DESTINATION include/aeon)

target_link_libraries(aeon ${SOX_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBRARIES} ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})
