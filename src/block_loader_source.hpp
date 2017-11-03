/*
 Copyright 2016 Intel(R) Nervana(TM)
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
#include <string>

#include "async_manager.hpp"
#include "buffer_batch.hpp"

namespace nervana
{
    class block_loader_source;

    typedef uint32_t source_uid_t;
}

class nervana::block_loader_source : public virtual async_manager_source<encoded_record_list>
{
public:
    virtual ~block_loader_source() {}
    virtual size_t       block_size() const  = 0;
    virtual size_t       block_count() const = 0;
    virtual source_uid_t get_uid() const     = 0;
};
