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

#include "batch_loader.hpp"
#include "batch_iterator.hpp"

class SequentialBatchIterator : public BatchIterator {
public:
    SequentialBatchIterator(std::shared_ptr<BatchLoader> loader, uint block_size);

    void read(BufferArray& dest);
    void reset();

private:
    std::shared_ptr<BatchLoader> _loader;
    uint _i;
    uint _block_size;
    uint _count;
};
