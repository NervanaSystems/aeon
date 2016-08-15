#pragma once

#include "provider_interface.hpp"
#include "etl_image_var.hpp"
#include "etl_localization.hpp"

namespace nervana {
    class image_localization : public provider_interface {
    public:
        image_localization(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

    private:
        image_var::config           image_config;
        localization::config        localization_config;

        image_var::extractor        image_extractor;
        image_var::transformer      image_transformer;
        image_var::loader           image_loader;
        image_var::param_factory    image_factory;

        localization::extractor     localization_extractor;
        localization::transformer   localization_transformer;
        localization::loader        localization_loader;
    };
}
