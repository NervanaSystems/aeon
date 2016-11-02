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

block_iterator_sequential::block_iterator_sequential(shared_ptr<block_loader> loader) :
    _loader(loader),
    _count(_loader->block_count()),
    _i(0)
{
    _loader->prefetch_block(_i);
}

void block_iterator_sequential::read(nervana::buffer_in_array& dest)
{
    // increment i before calling loadBlock so that if loadBlock throws an
    // exception, we've still incremented _i and the next call will request
    // the next i.  The policy here therefor is to skip blocks which throw
    // exceptions, there is no retry logic.
    auto i = _i;
    if (++_i == _count) {
        reset();
    }

    _loader->load_block(dest, i);
    _loader->prefetch_block(_i);
}

void block_iterator_sequential::reset()
{
    _i = 0;
}
