#pragma once

#include "provider_interface.hpp"
#include "etl_image.hpp"
#include "etl_pixel_mask.hpp"

namespace nervana {
    class image_pixelmask : public provider_interface {
    public:
        image_pixelmask(nlohmann::json js);

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

    private:
        image::config               image_config;
        image::config               target_config;
        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        pixel_mask::extractor       target_extractor;
        pixel_mask::transformer     target_transformer;
        image::loader               target_loader;
    };
}
