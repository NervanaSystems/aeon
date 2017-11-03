/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <random>

#include "json.hpp"
#include "interface.hpp"

namespace nervana
{
    namespace augment
    {
        namespace audio
        {
            class params;
            class param_factory;
        }
    }
}

class nervana::augment::audio::params : public interface::params
{
    friend class nervana::augment::audio::param_factory;

public:
    friend std::ostream& operator<<(std::ostream& out, const params& obj)
    {
        out << "add_noise               " << obj.add_noise << "\n";
        out << "noise_index             " << obj.noise_index << "\n";
        out << "noise_level             " << obj.noise_level << "\n";
        out << "noise_offset_fraction   " << obj.noise_offset_fraction << "\n";
        out << "time_scale_fraction     " << obj.time_scale_fraction << "\n";
        return out;
    }

    bool     add_noise;
    uint32_t noise_index;
    float    noise_level;
    float    noise_offset_fraction;
    float    time_scale_fraction;

private:
    params() {}
};

class nervana::augment::audio::param_factory : public json_configurable
{
public:
    param_factory(nlohmann::json config);
    std::shared_ptr<augment::audio::params> make_params() const;

    // This derived distribution gets filled by parsing add_noise_probability
    /** Probability of adding noise */
    mutable std::bernoulli_distribution add_noise{0.0f};
    /** Index into noise index file */
    mutable std::uniform_int_distribution<uint32_t> noise_index{0, UINT32_MAX};
    /** Offset from start of noise file */
    mutable std::uniform_real_distribution<float> noise_offset_fraction{0.0f, 1.0f};

    /** Simple linear time-warping */
    mutable std::uniform_real_distribution<float> time_scale_fraction{1.0f, 1.0f};

    /** How much noise to add (a value of 1 would be 0 dB SNR) */
    mutable std::uniform_real_distribution<float> noise_level{0.0f, 0.5f};

private:
    float add_noise_probability = 0.0f;

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        // ADD_SCALAR(type, mode::REQUIRED, [](const std::string& s) { return s == "audio"; }),

        // ADD_SCALAR(max_duration, mode::REQUIRED),
        // ADD_SCALAR(frame_stride, mode::REQUIRED),
        // ADD_SCALAR(frame_length, mode::REQUIRED),
        // ADD_SCALAR(num_cepstra, mode::OPTIONAL),
        // ADD_SCALAR(num_filters, mode::OPTIONAL),
        // ADD_SCALAR(output_type,
        //            mode::OPTIONAL,
        //            [](const std::string& v) { return output_type::is_valid_type(v); }),
        // ADD_SCALAR(feature_type, mode::OPTIONAL),
        // ADD_SCALAR(window_type, mode::OPTIONAL),
        // ADD_SCALAR(noise_root, mode::OPTIONAL),
        ADD_SCALAR(add_noise_probability, mode::OPTIONAL),

        ADD_DISTRIBUTION(time_scale_fraction,
                         mode::OPTIONAL,
                         [](decltype(time_scale_fraction) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(
            noise_level, mode::OPTIONAL, [](decltype(noise_level) v) { return v.a() <= v.b(); })};
};
