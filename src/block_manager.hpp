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
#include "block_loader_source.hpp"
#include "buffer_batch.hpp"
#include "block.hpp"

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
    block_manager(block_loader_source* file_loader,
                  size_t               block_size,
                  const std::string&   cache_root,
                  bool                 enable_shuffle);

    virtual ~block_manager() { finalize(); }
    encoded_record_list* filler() override;

    size_t record_count() const override { return m_block_size; }
    size_t elements_per_record() const override { return m_file_loader.elements_per_record(); }
private:
    static std::string create_cache_name(source_uid_t uid);
    static std::string create_cache_block_name(size_t block_number);

    static bool check_if_complete(const std::string& cache_dir);
    static void mark_cache_complete(const std::string& cache_dir);
    static bool take_ownership(const std::string& cache_dir, int& lock);
    static void release_ownership(const std::string& cache_dir, int lock);

    static const std::string m_owner_lock_filename;
    static const std::string m_cache_complete_filename;

    block_loader_source& m_file_loader;
    size_t               m_block_size;
    size_t               m_block_count;
    size_t               m_record_count;
    size_t               m_current_block_number;
    size_t               m_elements_per_record;
    const std::string    m_cache_root;
    bool                 m_cache_enabled;
    std::string          m_cache_dir;
    bool                 m_shuffle_enabled;
    source_uid_t         m_source_uid;
    int                  m_cache_lock = -1;
    size_t               m_cache_hit  = 0;
    size_t               m_cache_miss = 0;
    std::vector<size_t>  m_block_load_sequence;
    std::minstd_rand0    m_rnd;
};
