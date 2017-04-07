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

#include <math.h>
#include <sstream>

#include "block_loader.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

block_loader::block_loader(uint32_t block_size)
    : m_block_size(block_size)
{
}

uint32_t block_loader::block_size()
{
    return m_block_size;
}

uint32_t block_loader::block_count()
{
    return ceil((float)object_count() / (float)m_block_size);
}

void block_loader::prefetch_block(uint32_t block_num)
{
}
