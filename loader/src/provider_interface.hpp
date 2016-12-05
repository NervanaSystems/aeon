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

#include <memory>
#include "util.hpp"
#include "interface.hpp"
#include "buffer_in.hpp"
#include "buffer_out.hpp"

namespace nervana
{
    class provider_interface;
}

class nervana::provider_interface
{
public:
    virtual void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) = 0;
    virtual size_t get_input_count() const = 0;

    virtual void post_process(buffer_out_array& out_buf)
    {
    }

    const shape_type& get_output_shape(const std::string& name) const
    {
        auto it = m_output_shapes.find(name);
        if (it == m_output_shapes.end())
        {
            std::stringstream ss;
            ss << "key '" << name << "' not found";
            throw std::runtime_error(ss.str());
        }
        return it->second;
    }

    std::map<std::string,shape_type> get_output_shapes() const
    {
        return m_output_shapes;
    }

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
    std::map<std::string,shape_type> m_output_shapes;
    std::vector<std::string>         m_buffer_names;
};
