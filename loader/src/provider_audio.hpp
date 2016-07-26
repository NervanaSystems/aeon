#pragma once
#include "etl_char_map.hpp"
#include "etl_label.hpp"
#include "etl_audio.hpp"
#include "provider_interface.hpp"

namespace nervana {
    class transcribed_audio : public provider_interface {
    public:
        transcribed_audio(nlohmann::json js) :
            audio_config(js["audio"]),
            trans_config(js["transcript"]),
            audio_extractor(),
            audio_transformer(audio_config),
            audio_loader(audio_config),
            audio_factory(audio_config),
            trans_extractor(trans_config),
            trans_loader(trans_config)
        {
            num_inputs = 2;
            oshapes.push_back(audio_config.get_shape_type());
            oshapes.push_back(trans_config.get_shape_type());

            shape_type trans_length({1, 1}, output_type("uint32_t"));
            oshapes.push_back(trans_length);
        }

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override
        {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);

            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);
            char* length_out = out_buf[2]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process audio data
            auto audio_dec = audio_extractor.extract(datum_in.data(), datum_in.size());
            auto audio_params = audio_factory.make_params(audio_dec);
            audio_loader.load(datum_out, audio_transformer.transform(audio_params, audio_dec));

            // Process target data
            auto trans_dec = trans_extractor.extract(target_in.data(), target_in.size());
            trans_loader.load(target_out, trans_dec);

            // Save out the length
            uint32_t trans_length = trans_dec->get_length();
            pack_le(length_out, trans_length);
        }

        void post_process(buffer_out_array& out_buf) override
        {
            if (trans_config.pack_for_ctc) {
                auto num_items = out_buf[1]->getItemCount();
                char* dptr = out_buf[1]->data();
                uint32_t packed_len = 0;

                for (int i=0; i<num_items; i++) {
                    uint32_t len = unpack_le<uint32_t>(out_buf[2]->getItem(i));
                    memmove(dptr + packed_len, out_buf[1]->getItem(i), len);
                    packed_len += len;
                }
                memset(dptr + packed_len, 0, out_buf[1]->getSize() - packed_len);
            }
        }

        const std::unordered_map<char, uint8_t>& get_cmap() const {return trans_config.get_cmap();}

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

    class audio_classifier : public provider_interface {
    public:
        audio_classifier(nlohmann::json js) :
            audio_config(js["audio"]),
            label_config(js["label"]),
            audio_extractor(),
            audio_transformer(audio_config),
            audio_loader(audio_config),
            audio_factory(audio_config),
            label_extractor(label_config),
            label_loader(label_config)
        {
            num_inputs = 2;
            oshapes.push_back(audio_config.get_shape_type());
            oshapes.push_back(label_config.get_shape_type());
        }

        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) override
        {
            std::vector<char>& datum_in  = in_buf[0]->getItem(idx);
            std::vector<char>& target_in = in_buf[1]->getItem(idx);

            char* datum_out  = out_buf[0]->getItem(idx);
            char* target_out = out_buf[1]->getItem(idx);

            if (datum_in.size() == 0) {
                std::cout << "no data " << idx << std::endl;
                return;
            }

            // Process audio data
            auto audio_dec = audio_extractor.extract(datum_in.data(), datum_in.size());
            auto audio_params = audio_factory.make_params(audio_dec);
            audio_loader.load(datum_out, audio_transformer.transform(audio_params, audio_dec));

            // Process target data
            auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
            label_loader.load(target_out, label_dec);
        }

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
