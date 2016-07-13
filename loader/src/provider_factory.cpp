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
    } else if( mediaType == "localization" ) {
    } else {
        rc = nullptr;
        throw std::runtime_error("WTF??");
    }
    return rc;
}

// This is purely for peeking at a config structure without having to create a provider
std::shared_ptr<nervana::interface::config> nervana::config_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::interface::config> rc;
    if(!configJs["type"].is_string()) {
        throw std::invalid_argument("must have a property 'type'.  Options are: 'image' and 'label'.");
    }
    std::string mediaType = configJs["type"];
    if( mediaType == "image" ) {
        rc = std::make_shared<nervana::image::config>(configJs["config"]);
        printf("Made it here\n");
    } else if ( mediaType == "label" ) {
        rc = std::make_shared<nervana::label::config>(configJs["config"]);
    } else {
        rc = nullptr;
    }
    if (rc != nullptr) {
        if(!configJs["config"].is_object()) {
            throw std::invalid_argument("must have a property 'config' of type object.");
        }
    }
    return rc;
}
