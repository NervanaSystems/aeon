#pragma once
#include <memory>
#include "provider_interface.hpp"

namespace nervana {
    class provider_factory;
}

class nervana::provider_factory {
public:
    virtual ~provider_factory() {}

public:
    static std::shared_ptr<nervana::provider_interface> create(nlohmann::json configJs);
};

