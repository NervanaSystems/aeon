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

#include <memory>

#include "thread_pool.hpp"
#include "buffer_pool_in.hpp"
#include "batch_iterator.hpp"

namespace nervana
{
    class read_thread_pool;
}

/*
 * The read_thread_pool wraps BatchIterator in a thread an coordinates work
 * with other threads via locks on the output BufferPool `out`
 *
 */

class nervana::read_thread_pool : public thread_pool
{
public:
    read_thread_pool(const std::shared_ptr<nervana::buffer_pool_in>& out,
                     const std::shared_ptr<nervana::batch_iterator>& batch_iterator);

protected:
    virtual void work(int id) override;

private:
    read_thread_pool();
    read_thread_pool(const read_thread_pool&);
    std::shared_ptr<nervana::buffer_pool_in> m_out;
    std::shared_ptr<nervana::batch_iterator> m_batch_iterator;
};
