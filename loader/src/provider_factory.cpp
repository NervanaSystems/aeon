#include "provider_factory.hpp"
#include "provider_image_class.hpp"
#include "provider_audio.hpp"
#include "provider_video.hpp"

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
    } else if( mediaType == "audio,transcribe" ) {
        rc = make_shared<audio_transcriber>(configJs);
    } else if( mediaType == "audio,label" ) {
        rc = make_shared<audio_classifier>(configJs);
    } else if( mediaType == "audio" ) {
        rc = make_shared<audio_only>(configJs);
    } else if( mediaType == "image,localization" ) {
        rc = make_shared<localization_decoder>(configJs);
    } else if( mediaType == "image,pixelmask" ) {
        rc = make_shared<provider_pixel_mask>(configJs);
    } else if( mediaType == "image,boundingbox" ) {
        rc = make_shared<bbox_provider>(configJs);
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
