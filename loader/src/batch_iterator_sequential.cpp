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

#include "batch_iterator_sequential.hpp"

using namespace std;

BatchIteratorSequential::BatchIteratorSequential(shared_ptr<BatchLoader> loader)
: _loader(loader), _count(_loader->blockCount()), _i(0)
{
}

void BatchIteratorSequential::read(buffer_in_array& dest)
{
    _loader->loadBlock(dest, _i);
    if (_i++ == _count) {
        reset();
    }
}

void BatchIteratorSequential::reset()
{
    _i = 0;
}
