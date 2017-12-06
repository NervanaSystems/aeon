/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this nds except in compliance with the License.
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

#include "block_loader_source.hpp"
#include "manifest_nds.hpp"
#include "buffer_batch.hpp"

/* block_loader_nds
 *
 * Loads blocks from nds.
 *
 */

namespace nervana
{
    class block_loader_nds;
}

class nervana::block_loader_nds : public block_loader_source,
                                  public async_manager<encoded_record_list, encoded_record_list>
{
public:
    block_loader_nds(manifest_nds* manifest, size_t block_size);

    virtual ~block_loader_nds() { finalize(); }
    encoded_record_list* filler() override;

    size_t       record_count() const override { return m_record_count; }
    size_t       block_size() const override { return m_block_size; }
    size_t       block_count() const override { return m_block_count; }
    size_t       elements_per_record() const override { return m_elements_per_record; }
    source_uid_t get_uid() const override { return 0; }
private:
    size_t m_block_size;
    size_t m_block_count;
    size_t m_record_count;
    size_t m_elements_per_record;
};
