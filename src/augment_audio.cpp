/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
#ifndef PYTHON_PLUGIN
        if (js.find("plugin_filename") != js.end())
            WARN << "Detected `plugin_filename` in augmentation config, but aeon was not compiled "
                    "with "
                    "PYTHON_PLUGIN flag. Recompile with PYTHON_PLUGIN flag in order to use plugins."
                 << std::endl;
        if (js.find("plugin_params") != js.end())
            WARN << "Detected `plugin_params` in augmentation config, but aeon was not compiled "
                    "with "
                    "PYTHON_PLUGIN flag. Recompile with PYTHON_PLUGIN flag in order to use plugins."
                 << std::endl;
#endif
    }
}

shared_ptr<augment::audio::params> augment::audio::param_factory::make_params() const
{
    auto audio_stgs = shared_ptr<augment::audio::params>(new augment::audio::params());

#ifdef PYTHON_PLUGIN
    if (!plugin_filename.empty())
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (user_plugin_map.find(this_thread::get_id()) == user_plugin_map.end())
        {
            user_plugin_map[std::this_thread::get_id()] =
                std::make_shared<plugin>(plugin_filename, plugin_params.dump());
        }
        audio_stgs->user_plugin = user_plugin_map[this_thread::get_id()];

        if (audio_stgs->user_plugin)
            audio_stgs->user_plugin->prepare();
    }
    else
    {
        audio_stgs->user_plugin.reset();
    }
#endif

    auto& random = get_thread_local_random_engine();

    audio_stgs->add_noise             = add_noise(random);
    audio_stgs->noise_index           = noise_index(random);
    audio_stgs->noise_level           = noise_level(random);
    audio_stgs->noise_offset_fraction = noise_offset_fraction(random);
    audio_stgs->time_scale_fraction   = time_scale_fraction(random);

    return audio_stgs;
}
