#pragma once

#include "provider_interface.hpp"
#include "etl_boundingbox.hpp"
#include "etl_image.hpp"

namespace nervana {
    class image_boundingbox : public provider_interface {
    public:
        image_boundingbox(nlohmann::json js);
        virtual ~image_boundingbox() {}

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);
    private:
        image_boundingbox() = delete;
        image::config               image_config;
        boundingbox::config         bbox_config;

        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        boundingbox::extractor      bbox_extractor;
        boundingbox::transformer    bbox_transformer;
        boundingbox::loader         bbox_loader;

        std::default_random_engine  _r_eng;
    };
}
