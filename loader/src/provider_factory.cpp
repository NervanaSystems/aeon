#include "loader.hpp"
#include "api.hpp"
#include "provider_factory.hpp"

std::shared_ptr<nervana::train_base> nervana::train_provider_factory::create(nlohmann::json configJs)
{
    std::shared_ptr<nervana::train_base> rc;
    std::string mediaType = configJs["media"];
    std::cout << "media type " << mediaType << std::endl;
    if( mediaType == "image" ) {
        rc = std::make_shared<nervana::image_decoder>(configJs);
    } else {
        rc = nullptr;
    }
    return rc;
}
