#include "provider_factory.hpp"

std::shared_ptr<nervana::provider_interface> nervana::train_provider_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::provider_interface> rc;
    if(!configJs["media"].is_string()) {
        throw std::invalid_argument("must have a property 'media' with type string.  Options are: 'image_label'' and 'localization'.");
    }
    std::string mediaType = configJs["media"];
    if( mediaType == "image_label" ) {
        rc = std::make_shared<nervana::image_decoder>(configJs);
    } else if( mediaType == "audio_transcript" ) {
        rc = std::make_shared<nervana::transcribed_audio>(configJs);
    } else if( mediaType == "localization" ) {
    } else {
        rc = nullptr;
        throw std::runtime_error("WTF??");
    }
    return rc;
}
