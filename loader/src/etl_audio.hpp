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

#include "etl_interface.hpp"
#include "params.hpp"
#include "media.hpp"
#include "codec.hpp"
#include "specgram.hpp"
#include "noise_clips.hpp"

class Specgram;
class NoiseClips;
class Codec;

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

    class audio::params : public nervana::params {
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

        //          Independent      variables     and required
        uint32_t    freq_steps;
        uint32_t    max_duration_ms;

        //          Independent      variables     and optional
        int32_t     seed             {0};          //  Default  is to seed deterministically
        uint32_t    num_cepstra      {40};
        uint32_t    num_filters      {64};
        std::string type_string      {"uint8_t"};
        std::string feature_type     {"specgram"};
        std::string window_type      {"hann"};
        std::string noise_index_file {};

        uint32_t    sample_freq_hz   {16000};
        uint32_t    frame_stride_ms  {10};
        uint32_t    frame_length_ms  {25};

        //          Dependent        variables
        uint32_t    time_steps;
        uint32_t    frame_stride_tn;
        uint32_t    frame_length_tn;
        uint32_t    max_duration_tn;

        std::uniform_real_distribution<float>    time_scale_fraction   {1.0f, 1.0f};
        std::bernoulli_distribution              add_noise             {0.0f};
        std::uniform_real_distribution<uint32_t> noise_index           {0,    UINT32_MAX};
        std::uniform_real_distribution<float>    noise_level           {0.0f, 0.5f};
        std::uniform_real_distribution<float>    noise_offset_fraction {0.0f, 1.0f};

        bool audio::config::set_config(nlohmann::json js) override {
            parse_req(time_steps,      "time_steps",      js);
            parse_req(freq_steps,      "freq_steps",      js);
            parse_req(max_duration_ms, "max_duration_ms", js);

            parse_opt(sample_freq_hz,  "sample_freq_hz",  js);
            parse_opt(frame_stride_ms, "frame_stride_ms", js);
            parse_opt(frame_length_ms, "frame_length_ms", js);
            parse_opt(num_cepstra,     "num_cepstra",     js);
            parse_opt(num_filters,     "num_filters",     js);
            parse_opt(window_type,     "window_type",     js);
            parse_opt(feature_type,    "feature_type",    js);

            parse_opt(seed,            "seed",            js);
            parse_opt(type_string,     "type_string",     js);

            auto dist_params = js["distribution"];
            parse_dist(time_scale_fraction,   "time_scale_fraction",   dist_params);
            parse_dist(add_noise,             "add_noise",             dist_params);
            parse_dist(noise_index,           "noise_index",           dist_params);
            parse_dist(noise_level,           "noise_level",           dist_params);
            parse_dist(noise_offset_fraction, "noise_offset_fraction", dist_params);


            // Now fill in derived variables
            frame_stride_tn = frame_stride_ms * sample_freq_hz / 1000;
            frame_length_tn = frame_length_ms * sample_freq_hz / 1000;
            max_duration_tn = max_duration_ms * sample_freq_hz / 1000;

            time_steps      = (((max_duration_tn) - frame_length_tn) / frame_stride_tn);

            otype = nervana::output_type(type_string);
            if (type_string != "uint8_t") {
                throw std::runtime_error("Invalid load type for audio " + type_string);
            }
            shape = std::vector<uint32_t> {1, freq_steps, time_steps};
        }

        bool validate() {
            bool result = true;

            result &= frame_stride_ms != 0;
            result &= time_steps == (((max_duration_tn) - frame_length_tn) / frame_stride_tn) + 1

            return result;
        }
    };



    class audio::param_factory : public interface::param_factory<audio::decoded, audio::params> {
    public:
        param_factory(std::shared_ptr<audio::config> cfg) : _cfg{cfg}
        {
            // A positive provided seed means to run deterministic with that seed
            if (_cfg->seed >= 0) {
                _dre.seed((uint32_t) _cfg->seed);
            } else {
                _dre.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }
        ~param_factory() {}

        std::shared_ptr<audio::params> make_params(std::shared_ptr<const audio::decoded> input);
    private:
        std::shared_ptr<audio::config> _cfg;
        std::default_random_engine     _dre {0};
    };




    class audio::decoded : public decoded_media {
    public:
        decoded(std::shared_ptr<RawMedia> raw) : time_rep(raw) {}
        MediaType get_type() override { return MediaType::AUDIO; }
        size_t getSize() { return time_rep->numSamples(); }

        std::shared_ptr<RawMedia> get_time_data() { return time_rep; }
        cv::Mat &get_freq_data() { return freq_rep; }

    protected:
        std::shared_ptr<RawMedia> time_rep {nullptr};
        cv::Mat                   freq_rep {};
    };




    class audio::extractor : public interface::extractor<audio::decoded> {
    public:
        extractor(std::shared_ptr<const audio::config>)
        {
            _codec = std::make_shared<Codec>(MediaType::AUDIO);
            avcodec_register_all();
        }

        ~extractor() { _codec = nullptr; }

        std::shared_ptr<audio::decoded> extract(const char*, int) override;
    private:
        std::shared_ptr<Codec> _codec {nullptr};
    };





    class audio::transformer : public interface::transformer<audio::decoded, audio::params> {
    public:
        transformer(std::shared_ptr<const audio::config>);
        ~transformer();
        std::shared_ptr<audio::decoded> transform(
                                        std::shared_ptr<audio::params>,
                                        std::shared_ptr<audio::decoded>) override;
    private:
        std::shared_ptr<Specgram>   _specmaker  {nullptr};
        std::shared_ptr<NoiseClips> _noisemaker {nullptr};
    };
}
