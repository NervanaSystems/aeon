/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "provider_factory.hpp"
#include "provider_image_boundingbox.hpp"
#include "provider_image_classifier.hpp"
#include "provider_image_localization.hpp"
#include "provider_image_only.hpp"
#include "provider_image_pixelmask.hpp"
#include "provider_audio_classifier.hpp"
#include "provider_audio_only.hpp"
#include "provider_audio_transcriber.hpp"
#include "provider_video_classifier.hpp"
#include "provider_video_only.hpp"
#include "provider_image_stereo.hpp"

#include <sstream>

using namespace std;

std::shared_ptr<nervana::provider_interface> nervana::provider_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::provider_interface> rc;
    if(!configJs["type"].is_string()) {
        throw std::invalid_argument("must have a property 'type' with type string.");
    }
    std::string mediaType = configJs["type"];

    if( mediaType == "image,label" ) {
        rc = make_shared<image_classifier>(configJs);
    } else if( mediaType == "image" ) {
        rc = make_shared<image_only>(configJs);
    } else if( mediaType == "audio,transcription" ) {
        rc = make_shared<audio_transcriber>(configJs);
    } else if( mediaType == "audio,label" ) {
        rc = make_shared<audio_classifier>(configJs);
    } else if( mediaType == "audio" ) {
        rc = make_shared<audio_only>(configJs);
    } else if( mediaType == "image,localization" ) {
        rc = make_shared<image_localization>(configJs);
    } else if( mediaType == "image,pixelmask" ) {
        rc = make_shared<image_pixelmask>(configJs);
    } else if( mediaType == "image,boundingbox" ) {
        rc = make_shared<image_boundingbox>(configJs);
    } else if( mediaType == "stereo_image,blob" ) {
        rc = make_shared<image_stereo_blob>(configJs);
    } else if( mediaType == "video,label" ) {
        rc = make_shared<video_classifier>(configJs);
    } else if( mediaType == "video" ) {
        rc = make_shared<video_only>(configJs);
    } else {
        rc = nullptr;
        stringstream ss;
        ss << "provider type '" << mediaType << "' is not supported.";
        throw std::runtime_error(ss.str());
    }
    return rc;
}
