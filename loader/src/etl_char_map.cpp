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

#include "etl_char_map.hpp"

using namespace std;
using namespace nervana;

std::shared_ptr<char_map::decoded> char_map::extractor::extract(const char* in_array, int in_sz)
{
    uint32_t        nvalid = std::min((uint32_t)in_sz, _max_length);
    string          transcript(in_array, nvalid);
    vector<uint8_t> char_ints((vector<uint8_t>::size_type)_max_length, (uint8_t)0);

    uint32_t j = 0;
    for (uint32_t i = 0; i < nvalid; i++)
    {
        auto l = _cmap.find(std::toupper(transcript[i]));
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

void char_map::loader::load(const vector<void*>& outlist, std::shared_ptr<char_map::decoded> dc)
{
    char* outbuf = (char*)outlist[0];
    for (auto c : dc->get_data())
    {
        *(outbuf++) = c;
    }
}
