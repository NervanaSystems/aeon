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

#include <vector>
#include <cstdlib>

namespace nervana
{
    class block_info
    {
    public:
        block_info(size_t start, size_t count)
            : m_start{start}
            , m_count{count}
        {
        }
        size_t start() const { return m_start; }
        size_t count() const { return m_count; }
        size_t end() const { return m_start + m_count; }
    private:
        size_t m_start;
        size_t m_count;
    };

    std::vector<block_info> generate_block_list(size_t record_count, size_t block_size);
}
