/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#ifdef __linux__
#endif

namespace nervana
{
    template <typename T, void (T::*process_func)(int index)>
    class thread_pool_queue;
}

template <typename T, void (T::*process_func)(int index)>
class nervana::thread_pool_queue
{
public:
    thread_pool_queue(const std::vector<int>& thread_affinity_map)
    {
    }

    ~thread_pool_queue()
    {
    }

    void run(T* worker, int task_count)
    {
    }
};
