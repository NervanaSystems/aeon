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

#include <string>
#include <vector>

#include "log.hpp"

namespace nervana
{
    class manifest
    {
    public:
        enum class element_t
        {
            FILE,
            BINARY,
            STRING,
            ASCII_INT,
            ASCII_FLOAT
        };

        manifest() {}
        virtual ~manifest() {}
        virtual std::string cache_id() = 0;
        virtual std::string version()  = 0;

        void set_block_load_sequence(const std::vector<std::pair<size_t, size_t>>& seq)
        {
            m_block_load_sequence = seq;
        }

    protected:
        std::vector<std::pair<size_t, size_t>> m_block_load_sequence;

        manifest(const manifest&) = default;
    };
}
