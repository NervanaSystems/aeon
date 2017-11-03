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

#include <memory>
#include <cstddef>

#include "util.hpp"
#include "interface.hpp"
#include "buffer_batch.hpp"
#include "augment_image.hpp"
#include "augment_audio.hpp"

namespace nervana
{
    class provider_interface;
}

class nervana::provider_interface
{
public:
    provider_interface(nlohmann::json js, size_t input_count)
        : m_js(js)
        , m_input_count(input_count)
    {
    }

    virtual ~provider_interface() {}
    virtual void provide(int                           idx,
                         nervana::encoded_record_list& in_buf,
                         nervana::fixed_buffer_map&    out_buf) const = 0;

    size_t       get_input_count() const { return m_input_count; }
    virtual void post_process(fixed_buffer_map& out_buf) {}
    const shape_type& get_output_shape(const std::string& name) const
    {
        auto it =
            std::find_if(m_output_shapes.begin(),
                         m_output_shapes.end(),
                         [&](decltype(*m_output_shapes.begin())& v) { return v.first == name; });
        if (it == m_output_shapes.end())
        {
            std::stringstream ss;
            ss << "key '" << name << "' not found";
            throw std::runtime_error(ss.str());
        }
        return it->second;
    }

    const std::vector<std::pair<std::string, shape_type>>& get_output_shapes() const
    {
        return m_output_shapes;
    }
    nlohmann::json                  get_config() { return m_js; }
    const std::vector<std::string>& get_buffer_names()
    {
        if (m_buffer_names.empty())
        {
            for (auto i : m_output_shapes)
            {
                m_buffer_names.push_back(i.first);
            }
        }
        return m_buffer_names;
    }

protected:
    std::vector<std::pair<std::string, shape_type>> m_output_shapes;
    std::vector<std::string> m_buffer_names;
    nlohmann::json           m_js;
    size_t                   m_input_count{0};
};
