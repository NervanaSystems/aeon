/*
 Copyright 2015 Nervana Systems Inc.
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

#include "batch_loader.hpp"
#include "batch_iterator.hpp"

// This batch iterator shuffles the order that macro blocks are used as
// well as shuffling the data in the buffers.
class ShuffledBatchIterator : public BatchIterator {
public:
    ShuffledBatchIterator(shared_ptr<BatchLoader> loader, uint block_size, uint seed);

    void read(BufferPair& dest);
    void reset();

protected:
    void shuffle();

private:
    std::minstd_rand0 _rand;
    shared_ptr<BatchLoader> _loader;
    vector<uint> _indices;
    vector<uint>::iterator _it;
    uint _block_size;
    uint _seed;
    uint _epoch;
};
