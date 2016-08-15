#pragma once

#include "provider_interface.hpp"
#include "etl_label.hpp"
#include "etl_audio.hpp"

namespace nervana {
    class audio_classifier : public provider_interface {
    public:
        audio_classifier(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override;

    private:
        audio::config               audio_config;
        label::config               label_config;
        audio::extractor            audio_extractor;
        audio::transformer          audio_transformer;
        audio::loader               audio_loader;
        audio::param_factory        audio_factory;

        label::extractor            label_extractor;
        label::loader               label_loader;
    };

}
