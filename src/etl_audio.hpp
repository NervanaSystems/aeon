/*
 Copyright 2016 Intel(R) Nervana(TM)
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
#include "specgram.hpp"
#include "util.hpp"

#include "noise_clips.hpp"
#include "augment_audio.hpp"

class noise_clips;

namespace nervana
{
    namespace audio
    {
        class config;
        class params;
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }
}

/**
* \brief Configuration for audio ETL
*
* An instantiation of this class controls the ETL of audio data into the
* target memory buffers from the source CPIO archives.
*
* **Extract**
* - Audio files should be stored in mono, PCM-16 .wav format. If needed, you should create an ingestion script to convert to this
* format (commonly using sox or ffmpeg)
*
* **Transform**
* - Audio can be transformed into three different representations: spectrogram, mel-frequency spectral coefficients, or
* mel-frequency cepstral coefficients
*
* **Load**
* - Just a simple loading into a memory buffer.
*/
class nervana::audio::config : public interface::config
{
public:
    std::string name;
    // We will use the following suffixes for units:
    // _tn for samples, _ms for milliseconds, _ss for seconds, _hz for freq in hz

    // Independent variables and required
    /** \defgroup Required */

    /** Maximum duration of any audio clip in units of "seconds", "milliseconds", or
    * "samples" (e.g. "4 seconds"). */
    std::string max_duration;
    /** Interval between consecutive frames in units of "seconds", "milliseconds", or
    * "samples" (e.g. .01 seconds). */
    std::string frame_stride;
    /** Duration of each frame in units of "seconds", "milliseconds", or "samples"
    * (e.g. .025 seconds). */
    std::string frame_length;

    // Independent variables and optional

    /** Number of cepstra to use (only for feature_type="mfcc") */
    uint32_t num_cepstra{40};
    /** Number of filters to use for mel-frequency transform (used in feature_type="mfsc") (Not used for mfcc?) */
    uint32_t num_filters{64};
    /** Input data type. Currently only "uint8_t" is supported. */
    std::string output_type{"uint8_t"};
    /** Feature space to represent audio. Options are "specgram" - Short-time fourier transform, "mfsc" - Mel-Frequency spectrogram,
     * "mfcc" - Mel-Frequency cepstral coefficients */
    std::string feature_type{"specgram"};
    /** Window type for spectrogram generation */
    std::string window_type{"hann"};

    std::string noise_index_file{};
    std::string noise_root{};

    /** Sample rate of input audio in hertz */
    uint32_t sample_freq_hz{16000};

    /** Simple linear time-warping */
    std::uniform_real_distribution<float> time_scale_fraction{1.0f, 1.0f};

    /** How much noise to add (a value of 1 would be 0 dB SNR) */
    std::uniform_real_distribution<float> noise_level{0.0f, 0.5f};

    // Dependent variables
    uint32_t time_steps, freq_steps;
    uint32_t max_duration_tn, frame_length_tn, frame_stride_tn;
    float    max_duration_ms, frame_length_ms, frame_stride_ms;

    // This derived distribution gets filled by parsing add_noise_probability
    /** Probability of adding noise */
    std::bernoulli_distribution add_noise{0.0f};
    /** Index into noise index file */
    std::uniform_int_distribution<uint32_t> noise_index{0, UINT32_MAX};
    /** Offset from start of noise file */
    std::uniform_real_distribution<float> noise_offset_fraction{0.0f, 1.0f};

    /** Whether to also output length of the buffer */
    bool emit_length = false;

    /** \brief Parses the configuration JSON
    */
    config(nlohmann::json js)
    {
        if (js.is_null())
        {
            throw std::runtime_error("missing audio config in json config");
        }

        for (auto& info : config_list)
        {
            info->parse(js);
        }
        verify_config("audio", config_list, js);

        // Now fill in derived variables
        parse_samples_or_seconds(max_duration, max_duration_ms, max_duration_tn);
        parse_samples_or_seconds(frame_length, frame_length_ms, frame_length_tn);
        parse_samples_or_seconds(frame_stride, frame_stride_ms, frame_stride_tn);

        time_steps = ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1;

        if (feature_type == "specgram")
        {
            freq_steps = frame_length_tn / 2 + 1;
        }
        else if (feature_type == "mfsc")
        {
            freq_steps = num_filters;
        }
        else if (feature_type == "mfcc")
        {
            freq_steps = num_cepstra;
        }
        else if (feature_type == "samples")
        {
            freq_steps = 1;
        }
        else
        {
            throw std::runtime_error("Unknown feature type " + feature_type);
        }

        if (feature_type == "samples")
        {
            if (output_type != "int16_t" && output_type != "float")
            {
                throw std::runtime_error("Invalid pload type for audio " + output_type);
            }
        }
        else
        {
            if (output_type != "uint8_t")
            {
                throw std::runtime_error("Invalid dload type for audio " + output_type);
            }
        }

        add_noise = std::bernoulli_distribution{add_noise_probability};
        add_shape_type({1, freq_steps, time_steps}, {"channels", "frequency", "time"}, output_type);
        if (emit_length)
        {
            add_shape_type({1}, "uint32_t");
        }

        validate();
    }

    void validate()
    {
        if (frame_stride_ms <= 0)
        {
            throw std::invalid_argument("frame_stride_ms <= 0");
        }
        if (feature_type == "samples")
        {
            if (frame_length_tn != 1 || frame_stride_tn != 1)
            {
                throw std::invalid_argument(
                    "frame and stride must both be 1 sample to use raw sample feature_type");
            }
        }
        if (time_steps != ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1)
        {
            throw std::invalid_argument(
                "time_steps != ((max_duration_tn - frame_length_tn) / frame_stride_tn) + 1");
        }
        if (noise_offset_fraction.param().a() < 0.0f)
        {
            throw std::invalid_argument("noise_offset_fraction.param().a() < 0.0f");
        }
        if (noise_offset_fraction.param().b() > 1.0f)
        {
            throw std::invalid_argument("noise_offset_fraction.param().b() > 1.0f");
        }
    }

private:
    config() {}
    float                                                          add_noise_probability = 0.0f;
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list           = {
        ADD_SCALAR(max_duration, mode::REQUIRED),
        ADD_SCALAR(frame_stride, mode::REQUIRED),
        ADD_SCALAR(frame_length, mode::REQUIRED),
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(num_cepstra, mode::OPTIONAL),
        ADD_SCALAR(num_filters, mode::OPTIONAL),
        ADD_SCALAR(output_type,
                   mode::OPTIONAL,
                   [](const std::string& v) { return output_type::is_valid_type(v); }),
        ADD_SCALAR(feature_type, mode::OPTIONAL),
        ADD_SCALAR(window_type, mode::OPTIONAL),
        ADD_SCALAR(noise_index_file, mode::OPTIONAL),
        ADD_SCALAR(noise_root, mode::OPTIONAL),
        ADD_SCALAR(add_noise_probability, mode::OPTIONAL),
        ADD_SCALAR(sample_freq_hz, mode::OPTIONAL),
        ADD_DISTRIBUTION(time_scale_fraction,
                         mode::OPTIONAL,
                         [](decltype(time_scale_fraction) v) { return v.a() <= v.b(); }),
        // ADD_DISTRIBUTION(noise_index, mode::OPTIONAL),
        ADD_DISTRIBUTION(
            noise_level, mode::OPTIONAL, [](decltype(noise_level) v) { return v.a() <= v.b(); }),
        ADD_SCALAR(emit_length, mode::OPTIONAL),
    };

    void parse_samples_or_seconds(const std::string& unit, float& ms, uint32_t& tn)
    {
        // There's got to be a better way to do this (json parsing doesn't allow getting
        // as heterogenous sequences)

        std::string::size_type sz;
        const float            unit_val  = std::stof(unit, &sz);
        std::string            unit_type = unit.substr(sz);

        if (unit_type.find("samples") != std::string::npos)
        {
            tn = unit_val;
            ms = tn * 1000.0f / sample_freq_hz;
        }
        else if (unit_type.find("milliseconds") != std::string::npos)
        {
            ms = unit_val;
            tn = ms * sample_freq_hz / 1000.0f;
        }
        else if (unit_type.find("seconds") != std::string::npos)
        {
            ms = unit_val * 1000.0f;
            tn = unit_val * sample_freq_hz;
        }
        else
        {
            throw std::runtime_error("Unknown time unit " + unit_type);
        }
    }
};

class nervana::audio::decoded : public interface::decoded_media
{
public:
    decoded(cv::Mat raw)
        : time_rep{raw}
    {
    }
    size_t   size() { return time_rep.rows; }
    cv::Mat& get_time_data() { return time_rep; }
    cv::Mat& get_freq_data() { return freq_rep; }
    uint32_t valid_frames{0};

protected:
    cv::Mat time_rep{};
    cv::Mat freq_rep{};
};

class nervana::audio::extractor : public interface::extractor<audio::decoded>
{
public:
    extractor() {}
    ~extractor() {}
    std::shared_ptr<audio::decoded> extract(const void*, size_t) const override;

private:
};

class nervana::audio::transformer
    : public interface::transformer<audio::decoded, augment::audio::params>
{
public:
    transformer(const audio::config& config);
    ~transformer();
    std::shared_ptr<audio::decoded> transform(std::shared_ptr<augment::audio::params>,
                                              std::shared_ptr<audio::decoded>) const override;

private:
    transformer() = delete;
    void scale_time(cv::Mat& img, float scale_fraction);

    std::shared_ptr<noise_clips> _noisemaker{nullptr};
    const audio::config&         _cfg;
    cv::Mat                      _window{};
    cv::Mat                      _filterbank{};
};

class nervana::audio::loader : public interface::loader<audio::decoded>
{
public:
    loader(const audio::config& cfg)
        : _cfg{cfg}
    {
    }
    ~loader() {}
    virtual void load(const std::vector<void*>&, std::shared_ptr<audio::decoded>) const override;

private:
    const audio::config& _cfg;
};
