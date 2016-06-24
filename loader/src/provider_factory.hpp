#pragma once
#include <memory>
#include "provider.hpp"
#include "provider_image_class.hpp"

namespace nervana {
    class train_provider_factory;
}

class nervana::train_provider_factory {
public:
    virtual ~train_provider_factory() {}

public:
    static std::shared_ptr<nervana::train_base> create(nlohmann::json configJs);
};
