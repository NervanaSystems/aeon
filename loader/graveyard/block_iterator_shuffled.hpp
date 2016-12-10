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
#include <random>
#include "block_loader.hpp"
#include "block_iterator.hpp"

namespace nervana
{
    class block_iterator_shuffled;
}

// This batch iterator shuffles the order that macro blocks are used as
// well as shuffling the data in the buffers.
class nervana::block_iterator_shuffled : public block_iterator
{
public:
    block_iterator_shuffled(std::shared_ptr<block_loader> loader);
    void read(nervana::buffer_in_array& dest) override;
    void reset() override;

protected:
    void shuffle();

private:
    std::minstd_rand0               m_rand;
    std::shared_ptr<block_loader>   m_loader;
    std::vector<uint32_t>           m_indices;
    std::vector<uint32_t>::iterator m_it;
    uint32_t                        m_epoch;
};
