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

#include "params.hpp"
#include "etl_interface.hpp"
#include "raw_media.hpp"
#include "codec.hpp"
#include "specgram.hpp"
#include "noise_clips.hpp"

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
        std::bernoulli_distribution              add_noise             {0.0f};
        std::uniform_int_distribution<uint32_t>  noise_index           {0,    UINT32_MAX};
        std::uniform_real_distribution<float>    noise_level           {0.0f, 0.5f};
        std::uniform_real_distribution<float>    noise_offset_fraction {0.0f, 1.0f};

        // Dependent variables
        uint32_t    time_steps, freq_steps;
        uint32_t    max_duration_tn, frame_length_tn, frame_stride_tn;
        float       max_duration_ms, frame_length_ms, frame_stride_ms;

        config(nlohmann::json js) {
            parse_value(max_duration, "max_duration", js, mode::REQUIRED);
            parse_value(frame_stride, "frame_stride", js, mode::REQUIRED);
            parse_value(frame_length, "frame_length", js, mode::REQUIRED);

            parse_value(sample_freq_hz,  "sample_freq_hz", js, mode::OPTIONAL);
            parse_value(num_cepstra,     "num_cepstra",    js, mode::OPTIONAL);
            parse_value(num_filters,     "num_filters",    js, mode::OPTIONAL);
            parse_value(window_type,     "window_type",    js, mode::OPTIONAL);
            parse_value(feature_type,    "feature_type",   js, mode::OPTIONAL);

            parse_value(seed,            "seed",           js, mode::OPTIONAL);
            parse_value(type_string,     "type_string",    js, mode::OPTIONAL);

            auto dist_params = js["distribution"];
            parse_dist(time_scale_fraction,   "time_scale_fraction",   dist_params);
            parse_dist(add_noise,             "add_noise",             dist_params);
            parse_dist(noise_index,           "noise_index",           dist_params);
            parse_dist(noise_level,           "noise_level",           dist_params);
            parse_dist(noise_offset_fraction, "noise_offset_fraction", dist_params);

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

            otype = nervana::output_type(type_string);
            if (type_string != "uint8_t") {
                throw std::runtime_error("Invalid load type for audio " + type_string);
            }
            shape = std::vector<uint32_t> {1, freq_steps, time_steps};
            validate();
        }

        bool validate() {
            bool result = true;

            result &= frame_stride_ms != 0;
            result &= time_steps == ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1;

            result &= noise_offset_fraction.param().a() >= 0.0f;
            result &= noise_offset_fraction.param().b() <= 1.0f;

            return result;
        }
    private:
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



    class audio::decoded : public decoded_media {
    public:
        decoded(std::shared_ptr<RawMedia> raw) : time_rep(raw) {}
        MediaType get_type() override { return MediaType::AUDIO; }
        size_t getSize() { return time_rep->numSamples(); }

        std::shared_ptr<RawMedia> get_time_data() { return time_rep; }
        cv::Mat& get_freq_data() { return freq_rep; }

    protected:
        std::shared_ptr<RawMedia> time_rep {nullptr};
        cv::Mat                   freq_rep {};
    };




    class audio::extractor : public interface::extractor<audio::decoded> {
    public:
        extractor()
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

        transformer(const audio::config& config);
        ~transformer();
        std::shared_ptr<audio::decoded> transform(
                                        std::shared_ptr<audio::params>,
                                        std::shared_ptr<audio::decoded>) override;
    private:
        transformer() = delete;
        void resize(cv::Mat& img, float fx);

        std::shared_ptr<NoiseClips>    _noisemaker {nullptr};
        const audio::config&           _cfg;
        cv::Mat                        _window     {};
        cv::Mat                        _filterbank {};
    };
}
