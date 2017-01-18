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

#include "block_iterator_sequential.hpp"

using namespace std;
using namespace nervana;

block_iterator_sequential::block_iterator_sequential(shared_ptr<block_loader> loader)
    : m_loader(loader)
    , m_count(m_loader->block_count())
    , m_i(0)
{
    m_loader->prefetch_block(m_i);
}

void block_iterator_sequential::read(nervana::buffer_in_array& dest)
{
    // increment i before calling loadBlock so that if loadBlock throws an
    // exception, we've still incremented m_i and the next call will request
    // the next i.  The policy here therefor is to skip blocks which throw
    // exceptions, there is no retry logic.
    auto i = m_i;
    if (++m_i == m_count)
    {
        reset();
    }

    m_loader->load_block(dest, i);
    m_loader->prefetch_block(m_i);
}

void block_iterator_sequential::reset()
{
    m_i = 0;
}
