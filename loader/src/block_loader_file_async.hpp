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
#include <string>
#include "async_manager.hpp"
#include "manifest_csv.hpp"
#include "buffer_batch.hpp"
#include "util.hpp"
#include "crc.hpp"

/* block_loader_file
 *
 * Loads blocks of files from a Manifest into a BufferPair.
 *
 */

namespace nervana
{
    class block_loader_file_async;
}

class nervana::block_loader_file_async : public nervana::async_manager<std::vector<std::string>, nervana::variable_buffer_array>
{
public:
    block_loader_file_async(nervana::manifest_csv* mfst, uint32_t block_size);
    virtual ~block_loader_file_async() { finalize(); }
    virtual size_t                          object_count() override { return m_block_size; }
    virtual nervana::variable_buffer_array* filler() override;
    uint32_t                                block_size() { return m_block_size; }
private:
    uint32_t m_block_size;
    size_t   m_elements_per_record;
};
