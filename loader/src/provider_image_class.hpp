#pragma once

#include "etl_label.hpp"
#include "etl_image.hpp"
#include "provider_interface.hpp"
#include "etl_localization.hpp"
#include "etl_pixel_mask.hpp"

namespace nervana {
    class image_only : public provider_interface {
    public:
        image_only(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

    private:
        image::config               image_config;
        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;
    };

    class image_classifier : public provider_interface {
    public:
        image_classifier(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

    private:
        image::config               image_config;
        label::config               label_config;
        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        label::extractor            label_extractor;
        label::loader               label_loader;
    };

    class localization_decoder : public provider_interface {
    public:
        localization_decoder(nlohmann::json js);
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

    class bbox_provider : public provider_interface {
    public:
        bbox_provider(nlohmann::json js);
        virtual ~bbox_provider() {}

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);
    private:
        bbox_provider() = delete;
        image::config               image_config;
        bbox::config                bbox_config;

        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        bbox::extractor             bbox_extractor;
        bbox::transformer           bbox_transformer;
        bbox::loader                bbox_loader;

        std::default_random_engine  _r_eng;
    };

    class provider_pixel_mask : public provider_interface {
    public:
        provider_pixel_mask(nlohmann::json js);

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
