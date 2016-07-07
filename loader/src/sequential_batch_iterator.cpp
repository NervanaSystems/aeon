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

#include "buffer.hpp"

#include "sequential_batch_iterator.hpp"

using namespace std;

SequentialBatchIterator::SequentialBatchIterator(shared_ptr<BatchLoader> loader, uint block_size)
    : _loader(loader), _block_size(block_size) {
    _count = _loader->blockCount(block_size);

    reset();
};

void SequentialBatchIterator::read(BufferArray& dest) {
    _loader->loadBlock(dest, _i, _block_size);

    ++_i;

    if(_i == _count) {
        reset();
    }
}

void SequentialBatchIterator::reset() {
    _i = 0;
}
