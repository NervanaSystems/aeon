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

#include <vector>
#include <algorithm>
#include <random>

#include "util.hpp"
#include "block_iterator_shuffled.hpp"

using namespace std;
using namespace nervana;

block_iterator_shuffled::block_iterator_shuffled(shared_ptr<block_loader> loader)
    : m_rand(get_global_random_seed())
    , m_loader(loader)
    , m_epoch(0)
{
    // fill indices with integers from  0 to _count.  indices can then be
    // shuffled and used to iterate randomly through the blocks.
    m_indices.resize(m_loader->block_count());
    iota(m_indices.begin(), m_indices.end(), 0);
    shuffle();
    m_it = m_indices.begin();
    m_loader->prefetch_block(*m_it);
}

void block_iterator_shuffled::shuffle()
{
    std::shuffle(m_indices.begin(), m_indices.end(), m_rand);
}

void block_iterator_shuffled::read(nervana::buffer_in_array& dest)
{
    m_loader->load_block(dest, *m_it);

    // shuffle the objects in BufferPair dest
    // seed the shuffle with the seed passed in the constructor + the _epoch
    // to ensure that the buffer shuffles are deterministic wrt the input seed.
    // HACK: pass the same seed to both shuffles to ensure that both buffers
    // are shuffled in the same order.

    for (auto d : dest)
    {
        d->shuffle(get_global_random_seed() + m_epoch);
    }

    if (++m_it == m_indices.end())
    {
        reset();
    }
    m_loader->prefetch_block(*m_it);
}

void block_iterator_shuffled::reset()
{
    shuffle();
    m_it = m_indices.begin();
    ++m_epoch;
}
