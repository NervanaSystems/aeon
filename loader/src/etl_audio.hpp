/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <vector>
#include <chrono>

#include "interface.hpp"
#include "wav_data.hpp"
#include "specgram.hpp"
#include "noise_clips.hpp"

class noise_clips;

namespace nervana {
    namespace audio {
        class config;
        class params;
        class decoded;

        // goes from config -> params
        class param_factory;

        class extractor;
        class transformer;
        class loader;
    }

    class audio::params : public interface::params {
        friend class audio::param_factory;
    public:
        void dump(std::ostream & = std::cout);

        bool     add_noise;
        uint32_t noise_index;
        float    noise_level;
        float    noise_offset_fraction;
        float    time_scale_fraction;
    private:
        params() {}
    };

    class audio::config : public interface::config {
    public:
        // We will use the following suffixes for units:
        // _tn for samples, _ms for milliseconds, _ss for seconds, _hz for freq in hz

        // Independent variables and required
        std::string  max_duration;
        std::string  frame_stride;
        std::string  frame_length;


        // Independent variables and optional
        int32_t     seed             {0};          //  Default  is to seed deterministically
        uint32_t    num_cepstra      {40};
        uint32_t    num_filters      {64};
        std::string type_string      {"uint8_t"};
        std::string feature_type     {"specgram"};
        std::string window_type      {"hann"};

        std::string noise_index_file {};

        uint32_t    sample_freq_hz   {16000};

        std::uniform_real_distribution<float>    time_scale_fraction   {1.0f, 1.0f};
        std::uniform_real_distribution<float>    noise_level           {0.0f, 0.5f};

        // Dependent variables
        uint32_t    time_steps, freq_steps;
        uint32_t    max_duration_tn, frame_length_tn, frame_stride_tn;
        float       max_duration_ms, frame_length_ms, frame_stride_ms;

        // This derived distribution gets filled by parsing add_noise_probability
        std::bernoulli_distribution              add_noise             {0.0f};
        std::uniform_int_distribution<uint32_t>  noise_index           {0,    UINT32_MAX};
        std::uniform_real_distribution<float>    noise_offset_fraction {0.0f, 1.0f};


        config(nlohmann::json js) {
            if(js.is_null()) {
                throw std::runtime_error("missing audio config in json config");
            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config("audio", config_list, js);

            // Now fill in derived variables
            parse_samples_or_seconds(max_duration, max_duration_ms, max_duration_tn);
            parse_samples_or_seconds(frame_length, frame_length_ms, frame_length_tn);
            parse_samples_or_seconds(frame_stride, frame_stride_ms, frame_stride_tn);

            time_steps = ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1;

            if (feature_type == "specgram") {
                freq_steps = frame_length_tn / 2  + 1;
            } else if (feature_type == "mfsc") {
                freq_steps = num_filters;
            } else if (feature_type == "mfcc") {
                freq_steps = num_cepstra;
            } else {
                throw std::runtime_error("Unknown feature type " + feature_type);
            }

            if (type_string != "uint8_t") {
                throw std::runtime_error("Invalid load type for audio " + type_string);
            }
            add_noise = std::bernoulli_distribution{add_noise_probability};
            add_shape_type({1, freq_steps, time_steps}, type_string);
            validate();
        }

        void validate() {
            if(frame_stride_ms <= 0) {
                throw std::invalid_argument("frame_stride_ms <= 0");
            }
            if(time_steps != ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1) {
                throw std::invalid_argument("time_steps != ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1");
            }
            if(noise_offset_fraction.param().a() < 0.0f) {
                throw std::invalid_argument("noise_offset_fraction.param().a() < 0.0f");
            }
            if(noise_offset_fraction.param().b() > 1.0f) {
                throw std::invalid_argument("noise_offset_fraction.param().b() > 1.0f");
            }
        }
    private:
        config(){}
        float add_noise_probability = 0.0f;
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(max_duration, mode::REQUIRED),
            ADD_SCALAR(frame_stride, mode::REQUIRED),
            ADD_SCALAR(frame_length, mode::REQUIRED),
            ADD_SCALAR(seed, mode::OPTIONAL),
            ADD_SCALAR(num_cepstra, mode::OPTIONAL),
            ADD_SCALAR(num_filters, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL),
            ADD_SCALAR(feature_type, mode::OPTIONAL),
            ADD_SCALAR(window_type, mode::OPTIONAL),
            ADD_SCALAR(noise_index_file, mode::OPTIONAL),
            ADD_SCALAR(add_noise_probability, mode::OPTIONAL),
            ADD_SCALAR(sample_freq_hz, mode::OPTIONAL),
            ADD_DISTRIBUTION(time_scale_fraction, mode::OPTIONAL),
            // ADD_DISTRIBUTION(noise_index, mode::OPTIONAL),
            ADD_DISTRIBUTION(noise_level, mode::OPTIONAL),
        };

        void parse_samples_or_seconds(const std::string &unit, float &ms, uint32_t &tn)
        {
            // There's got to be a better way to do this (json parsing doesn't allow getting
            // as heterogenous sequences)

            std::string::size_type sz;
            const float unit_val = std::stof(unit, &sz);
            std::string unit_type = unit.substr(sz);

            if (unit_type.find("samples") != std::string::npos) {
                tn = unit_val;
                ms = tn * 1000.0f / sample_freq_hz;
            } else if (unit_type.find("milliseconds") != std::string::npos) {
                ms = unit_val;
                tn = ms * sample_freq_hz / 1000.0f;
            } else if (unit_type.find("seconds") != std::string::npos) {
                ms = unit_val * 1000.0f;
                tn = unit_val * sample_freq_hz;
            } else {
                throw std::runtime_error("Unknown time unit " + unit_type);
            }
        }
    };



    class audio::param_factory : public interface::param_factory<audio::decoded, audio::params> {
    public:
        param_factory(audio::config& cfg) : _cfg{cfg}
        {
            // A positive provided seed means to run deterministic with that seed
            if (_cfg.seed >= 0) {
                _dre.seed((uint32_t) _cfg.seed);
            } else {
                _dre.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }
        ~param_factory() {}

        std::shared_ptr<audio::params> make_params(std::shared_ptr<const audio::decoded> input);
    private:
        audio::config& _cfg;
        std::default_random_engine     _dre {0};
    };



    class audio::decoded : public interface::decoded_media {
    public:
        decoded(std::shared_ptr<wav_data> raw) : time_rep(raw) {}
        size_t size() { return time_rep->nsamples(); }

        std::shared_ptr<wav_data> get_time_data() { return time_rep; }
        cv::Mat& get_freq_data() { return freq_rep; }
        uint32_t valid_frames {0};
    protected:
        std::shared_ptr<wav_data> time_rep {nullptr};
        cv::Mat                   freq_rep {};
    };




    class audio::extractor : public interface::extractor<audio::decoded> {
    public:
        extractor() {}
        ~extractor() {}

        std::shared_ptr<audio::decoded> extract(const char*, int) override;
    private:
    };



    class audio::transformer : public interface::transformer<audio::decoded, audio::params> {
    public:

        transformer(const audio::config& config);
        ~transformer();
        std::shared_ptr<audio::decoded> transform(
                                        std::shared_ptr<audio::params>,
                                        std::shared_ptr<audio::decoded>) override;
    private:
        transformer() = delete;
        void scale_time(cv::Mat& img, float scale_fraction);

        std::shared_ptr<noise_clips>    _noisemaker {nullptr};
        const audio::config&           _cfg;
        cv::Mat                        _window     {};
        cv::Mat                        _filterbank {};
    };



    class audio::loader : public interface::loader<audio::decoded> {
    public:
        loader(const audio::config& cfg) : _cfg{cfg} {}
        ~loader() {}
        virtual void load(const std::vector<void*>&, std::shared_ptr<audio::decoded>) override;

    private:
        const audio::config& _cfg;
    };

}
