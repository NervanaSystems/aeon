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

#include <vector>

namespace nervana
{
    class buffer_pool;
}

/* Base class buffer_pool only deals in exception handling for now */

class nervana::buffer_pool
{
protected:
    buffer_pool();
public:
    void write_exception(std::exception_ptr exception_ptr);
    void reraise_exception();

protected:
    void clear_exception();

    std::vector<std::exception_ptr> m_exceptions;
    int                             m_read_pos = 0;
    int                             m_write_pos = 0;
};
