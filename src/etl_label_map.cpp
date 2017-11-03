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

#include <sstream>
#include <iostream>
#include "etl_label_map.hpp"

using namespace std;
using namespace nervana::label_map;

config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw std::runtime_error("missing label_map config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("label_map", config_list, js);

    // Derived types
    auto otype = nervana::output_type(output_type);
    add_shape_type({otype.get_size()}, otype);

    if (output_type != "uint32_t")
    {
        throw std::runtime_error("Invalid load type for label map " + output_type);
    }
}

decoded::decoded()
{
}

extractor::extractor(const label_map::config& cfg)
{
    int index = 0;
    for (const string& label : cfg.class_names)
    {
        dictionary.insert({label, index++});
    }
}

shared_ptr<decoded> extractor::extract(const void* data, size_t size) const
{
    auto         rc = make_shared<decoded>();
    stringstream ss(string((const char*)data, size));
    string       label;
    while (ss >> label)
    {
        auto l = dictionary.find(label);
        if (l != dictionary.end())
        {
            // found label
            rc->labels.push_back(l->second);
        }
        else
        {
            // label not found in dictionary
            rc = nullptr;
            break;
        }
    }
    return rc;
}

transformer::transformer()
{
}

shared_ptr<decoded> transformer::transform(shared_ptr<params> pptr, shared_ptr<decoded> media) const
{
    return media;
}

loader::loader(const nervana::label_map::config& cfg)
    : max_label_count{cfg.max_label_count()}
{
}

void loader::load(const vector<void*>& data, shared_ptr<decoded> media) const
{
    int       i      = 0;
    uint32_t* data_p = (uint32_t*)data[0];
    for (; i < media->get_data().size() && i < max_label_count; i++)
    {
        data_p[i] = media->get_data()[i];
    }
    for (; i < max_label_count; i++)
    {
        data_p[i] = 0;
    }
}
