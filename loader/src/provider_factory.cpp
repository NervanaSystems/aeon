#include "provider_factory.hpp"

std::shared_ptr<nervana::train_base> nervana::train_provider_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::train_base> rc;
    std::string mediaType = configJs["media"];
    std::cout << "media type " << mediaType << std::endl;
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
    std::string mediaType = configJs["type"];
    if( mediaType == "image" ) {
        rc = std::make_shared<nervana::image::config>();
        printf("Made it here\n");
    } else if ( mediaType == "label" ) {
        rc = std::make_shared<nervana::label::config>();
    } else {
        rc = nullptr;
    }
    if (rc != nullptr) {
        rc->set_config(configJs["config"]);
    }
    return rc;
}
