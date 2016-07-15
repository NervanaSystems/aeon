#pragma once
#include <memory>
#include "provider_interface.hpp"
#include "provider_image_class.hpp"
#include "provider_audio.hpp"

namespace nervana {
    class train_provider_factory;
}

class nervana::train_provider_factory {
public:
    virtual ~train_provider_factory() {}

public:
    static std::shared_ptr<nervana::provider_interface> create(nlohmann::json configJs);
};

