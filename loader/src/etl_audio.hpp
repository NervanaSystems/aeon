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
#include "raw_media.hpp"
#include "codec.hpp"
#include "specgram.hpp"
#include "noise_clips.hpp"

class Specgram;
class NoiseClips;
class NoiseClipsState;
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

        int _width;
        int _height;

    private:
        params() {}
    };

    class audio::param_factory : public interface::param_factory<audio::decoded, audio::params> {
    public:
        param_factory(std::shared_ptr<audio::config> cfg,
                      std::default_random_engine& dre) : _cfg{cfg} {}
        virtual ~param_factory() {}

        std::shared_ptr<audio::params> make_params(std::shared_ptr<const decoded>);
    private:
        std::shared_ptr<audio::config> _cfg;
    };

    class audio::config : public json_config_parser {
    public:
        config(nlohmann::json js);

        int                         _randomSeed = 0;
        int                         _samplingFreq;
        int                         _clipDuration;
        float                       _randomScalePercent;
        int                         _numFilts;
        int                         _numCepstra;
        std::string                 _noiseIndexFile = "";
        std::string                 _noiseDir = "";
        int                         _windowSize;
        int                         _stride;
        int                         _width;
        int                         _height;
        int                         _window;
        int                         _feature;
        MediaType                   _mtype = MediaType::AUDIO;

    private:
        config() = delete;
    };

    class audio::decoded : public decoded_media {
    public:
        decoded(std::shared_ptr<RawMedia> raw);
        virtual ~decoded(){}
        MediaType get_type() override;

        size_t getSize();
    // protected:
        std::shared_ptr<RawMedia> _raw;
        vector<char> _buf;
        int _len;
    };

    class audio::extractor : public interface::extractor<audio::decoded> {
    public:
        extractor(std::shared_ptr<const audio::config>);
        virtual ~extractor();
        std::shared_ptr<audio::decoded> extract(const char*, int) override;
    private:
        extractor() = delete;
        Codec* _codec;
    };

    class audio::transformer : public interface::transformer<audio::decoded, audio::params> {
    public:
        transformer(std::shared_ptr<const audio::config>);
        virtual ~transformer();
        std::shared_ptr<audio::decoded> transform(
            std::shared_ptr<audio::params>,
            std::shared_ptr<audio::decoded>) override;
    private:
        transformer() = delete;
        Codec* _codec;
        Specgram* _specgram;
        NoiseClips* _noiseClips;
        NoiseClipsState* _state;
        cv::RNG _rng;
    };
}
