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

#include <cmath>

#include "block.hpp"

using namespace nervana;

std::vector<block_info> nervana::generate_block_list(size_t record_count, size_t block_size)
{
    size_t block_count = round((float)record_count / (float)block_size);
    block_count        = std::max<size_t>(block_count, 1);
    block_size         = ceil((float)record_count / (float)block_count);

    std::vector<block_info> rc;
    for (size_t block = 0; block < block_count; block++)
    {
        size_t sequence_start = block_size * block;
        size_t sequence_count = block_size;
        if (sequence_start + sequence_count > record_count)
        {
            sequence_count = record_count - sequence_start;
            rc.emplace_back(sequence_start, sequence_count);
            break;
        }
        rc.emplace_back(sequence_start, sequence_count);
    }

    return rc;
}
