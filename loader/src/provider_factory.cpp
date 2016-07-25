#include "provider_factory.hpp"

#include <sstream>

using namespace std;

std::shared_ptr<nervana::provider_interface> nervana::train_provider_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::provider_interface> rc;
    if(!configJs["type"].is_string()) {
        throw std::invalid_argument("must have a property 'type' with type string.  Options are: 'image,label' and 'localization'.");
    }
    std::string mediaType = configJs["type"];
    if( mediaType == "image,label" ) {
        rc = std::make_shared<nervana::image_decoder>(configJs);
    } else if( mediaType == "audio_transcript" ) {
        rc = std::make_shared<nervana::transcribed_audio>(configJs);
    } else if( mediaType == "localization" ) {
    } else {
        rc = nullptr;
        stringstream ss;
        ss << "provider type '" << mediaType << "' is not supported.";
        throw std::runtime_error(ss.str());
    }
    return rc;
}
