#pragma once

#include "etl_label.hpp"
#include "etl_image.hpp"
#include "provider_interface.hpp"
#include "etl_localization.hpp"
#include "etl_pixel_mask.hpp"

namespace nervana {
    class image_decoder : public provider_interface {
    public:
        image_decoder(nlohmann::json js) :
            image_config(js["data_config"]["config"]),
            label_config(js["target_config"]["config"]),
            image_extractor(image_config),
            image_transformer(image_config),
            image_loader(image_config),
            image_factory(image_config),
            label_extractor(label_config),
            label_transformer(label_config),
            label_loader(label_config)
        {
        }

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);
            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process image data
            auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
            auto image_params = image_factory.make_params(image_dec);
            image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

            // Process target data
            auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
            label_loader.load(target_out, label_transformer.transform(image_params, label_dec));
        }

    private:
        image::config               image_config;
        label::config               label_config;
        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        label::extractor            label_extractor;
        label::transformer          label_transformer;
        label::loader               label_loader;

        std::default_random_engine  _r_eng;
    };

    class localization_decoder : public provider_interface {
    public:
        localization_decoder(nlohmann::json js) :
            image_config(js["data_config"]["config"]),
            localization_config(js["target_config"]["config"]),
            image_extractor(image_config),
            image_transformer(image_config),
            image_loader(image_config),
            image_factory(image_config),
            localization_extractor(localization_config),
            localization_transformer(localization_config),
            localization_loader(localization_config)
        {
        }

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);

            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process image data
            auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
            auto image_params = image_factory.make_params(image_dec);
            image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

            // Process target data
            auto target_dec = localization_extractor.extract(target_in.data(), target_in.size());
            localization_loader.load(target_out, localization_transformer.transform(image_params, target_dec));
        }

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

        std::default_random_engine  _r_eng;
    };

    class bbox_provider : public provider_interface {
    public:
        bbox_provider(nlohmann::json js) :
            image_config(js["data_config"]["config"]),
            bbox_config(js["target_config"]["config"]),
            image_extractor(image_config),
            image_transformer(image_config),
            image_loader(image_config),
            image_factory(image_config),
            bbox_extractor(bbox_config),
            bbox_transformer(bbox_config),
            bbox_loader(bbox_config)
        {
        }

        virtual ~bbox_provider() {}

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);

            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process image data
            auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
            auto image_params = image_factory.make_params(image_dec);
            image_loader.load(datum_out, image_transformer.transform(image_params, image_dec));

            // Process target data
            auto target_dec = bbox_extractor.extract(target_in.data(), target_in.size());
            bbox_loader.load(target_out, bbox_transformer.transform(image_params, target_dec));
        }
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

    class pixel_mask_decoder : public provider_interface {
    public:
        pixel_mask_decoder(nlohmann::json js) :
            image_config(js["data_config"]["config"]),
            image_extractor(image_config),
            image_transformer(image_config),
            image_loader(image_config),
            image_factory(image_config),
            target_transformer(image_config)
        {
        }

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);
            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process image data
            auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
            auto image_params = image_factory.make_params(image_dec);
            auto image_transformed = image_transformer.transform(image_params, image_dec);
            image_loader.load(datum_out, image_transformed);

            // Process target data
            auto target_dec = image_extractor.extract(target_in.data(), target_in.size());
            auto target_transformed = target_transformer.transform(image_params, target_dec);
            image_loader.load(target_out, target_transformed);
        }

    private:
        image::config               image_config;
        image::extractor            image_extractor;
        image::transformer          image_transformer;
        image::loader               image_loader;
        image::param_factory        image_factory;

        pixel_mask::transformer     target_transformer;
    };
}
