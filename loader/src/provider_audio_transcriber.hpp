#pragma once

#include "provider_interface.hpp"
#include "etl_char_map.hpp"
#include "etl_audio.hpp"

namespace nervana {
    class audio_transcriber : public provider_interface {
    public:
        audio_transcriber(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override;
        void post_process(buffer_out_array& out_buf) override;
        const std::unordered_map<char, uint8_t>& get_cmap() const
        {
            return trans_config.get_cmap();
        }

    private:
        audio::config               audio_config;
        char_map::config            trans_config;
        audio::extractor            audio_extractor;
        audio::transformer          audio_transformer;
        audio::loader               audio_loader;
        audio::param_factory        audio_factory;

        char_map::extractor         trans_extractor;
        char_map::loader            trans_loader;
    };
}
