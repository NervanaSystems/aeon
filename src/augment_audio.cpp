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

#include "augment_audio.hpp"

using namespace std;
using namespace nervana;

nervana::augment::audio::param_factory::param_factory(nlohmann::json js)
{
    if (js.is_null() == false)
    {
        string type;
        auto   val = js.find("type");
        if (val == js.end())
        {
            throw std::invalid_argument("augmentation missing 'type'");
        }
        else
        {
            type = val->get<string>();
            js.erase(val);
        }

        if (type == "audio")
        {
            for (auto& info : config_list)
            {
                info->parse(js);
            }
            // verify_config("audio", config_list, js);

            add_noise = std::bernoulli_distribution{add_noise_probability};
            // validate();
        }
    }
}

shared_ptr<augment::audio::params> augment::audio::param_factory::make_params() const
{
    auto audio_stgs = shared_ptr<augment::audio::params>(new augment::audio::params());

    auto& random = get_thread_local_random_engine();

    audio_stgs->add_noise             = add_noise(random);
    audio_stgs->noise_index           = noise_index(random);
    audio_stgs->noise_level           = noise_level(random);
    audio_stgs->noise_offset_fraction = noise_offset_fraction(random);
    audio_stgs->time_scale_fraction   = time_scale_fraction(random);

    return audio_stgs;
}
