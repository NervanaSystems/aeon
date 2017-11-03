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

#include "etl_char_map.hpp"

using namespace std;
using namespace nervana;

nervana::char_map::config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw std::runtime_error("missing char_map config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("char_map", config_list, js);

    // Now fill in derived
    add_shape_type({1, max_length}, {"character", "sequence"}, output_type);

    if (emit_length)
    {
        add_shape_type({1}, "uint32_t");
    }

    // set locale to operate on UTF8 input characters
    setlocale(LC_CTYPE, "");

    uint32_t index = 0;
    walphabet      = to_wstring(alphabet);
    for (auto c : walphabet)
    {
        _cmap.insert({std::towupper(c), index++});
    }

    validate();
}

std::shared_ptr<char_map::decoded> char_map::extractor::extract(const void* in_array,
                                                                size_t      in_sz) const
{
    uint32_t         nvalid = std::min((uint32_t)in_sz, _max_length);
    string           transcript((const char*)in_array);
    wstring          wtranscript = to_wstring(transcript, nvalid);
    vector<uint32_t> char_ints(_max_length, 0);

    uint32_t j = 0;
    for (uint32_t i = 0; i < nvalid; i++)
    {
        auto l = _cmap.find(std::towupper(wtranscript[i]));
        if (l == _cmap.end())
        {
            if (_unknown_value > 0)
            {
                char_ints[j++] = _unknown_value;
                continue;
            }
            else
            {
                continue;
            }
        }
        char_ints[j++] = l->second;
    }
    auto rc = make_shared<char_map::decoded>(char_ints, nvalid);
    return rc;
}

void char_map::loader::load(const vector<void*>&               outlist,
                            std::shared_ptr<char_map::decoded> dc) const
{
    wchar_t* outbuf = (wchar_t*)outlist[0];
    for (auto c : dc->get_data())
    {
        *(outbuf++) = c;
    }
    if (_emit_length)
    {
        uint32_t* length_buf = (uint32_t*)outlist[1];
        *length_buf          = dc->get_length();
    }
}
