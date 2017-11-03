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

#include <string>

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "block.hpp"
#include "block_loader_source.hpp"
#include "cache_system.hpp"

/* block_manager
 *
 * Reads files from the manifest and optionally caches and shuffles them.
 *
 */

namespace nervana
{
    class block_manager;
}

class nervana::block_manager : public async_manager<encoded_record_list, encoded_record_list>
{
public:
    block_manager(std::shared_ptr<block_loader_source> file_loader,
                  size_t                               block_size,
                  const std::string&                   cache_root,
                  bool                                 enable_shuffle,
                  uint32_t                             seed = 0);

    virtual ~block_manager() { finalize(); }
    encoded_record_list* filler() override;

    virtual void initialize() override;

    size_t record_count() const override { return m_block_size; }
    size_t elements_per_record() const override { return m_elements_per_record; }
private:
    std::unique_ptr<cache_system> m_cache;
    size_t                        m_current_block_number;
    size_t                        m_block_size;
    size_t                        m_block_count;
    size_t                        m_record_count;
    size_t                        m_elements_per_record;
};
