#pragma once

#include "provider.hpp"
#include "etl_label.hpp"
#include "etl_image.hpp"

namespace nervana {
    namespace image {
        class randomizing_provider;
    }
    namespace label {
        class int_provider;
    }

    class image::randomizing_provider : public provider<image::decoded, image::params> {
    public:
        randomizing_provider(nlohmann::json js, int seed=0)
        {
            _cfg         = std::make_shared<image::config>();
            _cfg->set_config(js);   // TODO: check return value
            _r_eng       = std::default_random_engine(seed);
            _extractor   = std::make_shared<image::extractor>(_cfg);
            _transformer = std::make_shared<image::transformer>(_cfg);
            _loader      = std::make_shared<image::loader>(_cfg);
            _factory     = std::make_shared<image::param_factory>(_cfg, _r_eng);
        }
        ~randomizing_provider() {}

    private:
        std::default_random_engine     _r_eng;
        std::shared_ptr<image::config> _cfg;
    };

    class label::int_provider : public provider<label::decoded, nervana::params> {
    public:
        int_provider(nlohmann::json js)
        {
            auto cfg     = std::make_shared<label::config>();
            cfg->set_config(js);
            _extractor   = std::make_shared<label::extractor>(cfg);
            _transformer = std::make_shared<label::transformer>();
            _loader      = std::make_shared<label::loader>();
            _factory     = nullptr;
        }
        ~int_provider() {}
    };

    class image_decoder : public train_provider<image::randomizing_provider, label::int_provider> {
    public:
        image_decoder(nlohmann::json js, int seed=0)
        {
            _dprov = std::make_shared<image::randomizing_provider>(js["data_config"], seed);
            _tprov = std::make_shared<label::int_provider>(js["target_config"]);
        }
    };
}
