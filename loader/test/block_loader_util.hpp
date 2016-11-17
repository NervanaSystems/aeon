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

#include "block_loader.hpp"

// mock alphabet block loader for use in tests
class block_loader_alphabet : public nervana::block_loader
{
public:
    block_loader_alphabet(uint32_t block_size);
    void load_block(nervana::buffer_in_array& dest, uint32_t block_num) override;
    uint32_t object_count() override { return 26 * m_block_size; }
};

// Random block loader used for tests
class block_loader_random : public nervana::block_loader
{
public:
    block_loader_random(uint32_t block_size)
        : block_loader(block_size)
    {
    }
    void load_block(nervana::buffer_in_array& dest, uint32_t block_num) override;
    uint32_t           object_count() override { return 10; } // Not really correct, but unused for tests
    static std::string randomString();
};
