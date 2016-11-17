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

#include <sstream>

#include "block_loader_util.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

block_loader_alphabet::block_loader_alphabet(uint32_t block_size) :
    block_loader(block_size)
{
    affirm(block_size < 26, "block_loader_alphabet block_size must be < 26");
}

void block_loader_alphabet::load_block(buffer_in_array &dest, uint32_t block_num)
{
    // load BufferPair with strings.
    // block_num 0: 'Aa', 'Ab', 'Ac'
    // block_num 1: 'Ba', 'Bb', 'Bc'
    // ...
    affirm(block_num < 26, "block_num must be < 26");

    for(uint32_t i = 0; i < m_block_size; ++i) {
        stringstream ss;
        ss << (char)('A' + block_num);
        ss << (char)('a' + i);
        string s = ss.str();

        for (auto d: dest) {
            d->add_item(vector<char>(s.begin(), s.end()));
        }
    }
}

void block_loader_random::load_block(buffer_in_array &dest, uint32_t block_num)
{
    // load BufferPair with random bytes
    std::random_device engine;

    string object_string = randomString();
    vector<char> obj(object_string.begin(), object_string.end());
    dest[0]->add_item(obj);

    string target = randomString();
    vector<char> tgt(target.begin(), target.end());
    dest[1]->add_item(tgt);
}

string block_loader_random::randomString()
{
    stringstream s;
    std::random_device engine;
    uint32_t x = engine();
    s << x;
    return s.str();
}
