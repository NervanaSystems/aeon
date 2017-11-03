/*
 Copyright 2017 Intel(R) Nervana(TM)
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
#include <mutex>

#include "buffer_batch.hpp"
#include "block_loader_source.hpp"

namespace nervana
{
    class cache_system;
}

class nervana::cache_system
{
public:
    cache_system(source_uid_t       uid,
                 size_t             block_count,
                 size_t             elements_per_record,
                 const std::string& cache_root,
                 bool               shuffle_enabled,
                 uint32_t           seed = 0);
    ~cache_system();
    void load_block(encoded_record_list& buffer);
    void store_block(const encoded_record_list& buffer);
    bool is_complete() { return m_stage == complete; }
    bool is_ownership() { return m_stage == ownership; }
    void try_get_access();
    void restart();

private:
    enum stages
    {
        complete,
        ownership,
        blocked
    } m_stage;
    static const std::string m_owner_lock_filename;
    static const std::string m_cache_complete_filename;
    size_t                   m_block_count;
    std::vector<size_t>      m_block_load_sequence;
    const std::string        m_cache_root;
    std::string              m_cache_dir;
    bool                     m_shuffle_enabled;
    size_t                   m_elements_per_record;
    size_t                   m_current_block_number;
    int                      m_cache_lock = -1;
    std::minstd_rand0        m_random;

    static std::mutex m_mutex;

    bool check_if_complete(const std::string& cache_dir);
    void mark_cache_complete(const std::string& cache_dir);
    bool take_ownership(const std::string& cache_dir, int& lock);
    void release_ownership(const std::string& cache_dir, int lock);
    std::string create_cache_name(source_uid_t uid);
    std::string create_cache_block_name(size_t block_number);
};
