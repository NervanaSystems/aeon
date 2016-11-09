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

#include "buffer_pool.hpp"

using namespace nervana;

buffer_pool::buffer_pool()
{
    _exceptions.push_back(nullptr);
    _exceptions.push_back(nullptr);
}

void buffer_pool::write_exception(std::exception_ptr exception_ptr)
{
    _exceptions[_writePos] = exception_ptr;
}

void buffer_pool::clear_exception()
{
    _exceptions[_writePos] = nullptr;
}

void buffer_pool::reraise_exception()
{
    if(auto e = _exceptions[_readPos]) {
        clear_exception();
        std::rethrow_exception(e);
    }
}
