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

#include "block_loader_file.hpp"

/* block_loader_cpio_cache
 *
 * caches loadBlock function in block_loader out to cpio files in
 * `cacheDir`.
 *
 * The cache_id is used to unquely identify a particular dataset and the version
 * is used to help invalidate old versions of the same dataset.  If a cache is
 * created with the same cache_id as an existing cache, but a different version,
 * old version is deleted.
 */

namespace nervana
{
    class block_loader_cpio_cache;
}

class nervana::block_loader_cpio_cache : public block_loader
{
public:
    block_loader_cpio_cache(const std::string& rootCacheDir, const std::string& cache_id, const std::string& version,
                            std::shared_ptr<block_loader> loader);

    void load_block(nervana::buffer_in_array& dest, uint32_t block_num) override;
    void prefetch_block(uint32_t block_num) override;
    uint32_t object_count() override;

private:
    bool load_block_from_cache(nervana::buffer_in_array& dest, uint32_t block_num);
    void write_block_to_cache(nervana::buffer_in_array& dest, uint32_t block_num);
    std::string block_filename(uint32_t block_num);

    void invalidate_old_cache(const std::string& rootCacheDir, const std::string& cache_id, const std::string& version);
    bool filename_holds_invalid_cache(const std::string& filename, const std::string& cache_id, const std::string& version);

    bool check_if_complete();
    void mark_cache_complete();
    bool take_ownership();
    void release_ownership();

    const std::string m_owner_lock_filename     = "caching_in_progress";
    const std::string m_cache_complete_filename = "cache_complete";

    std::string                   m_cache_dir;
    std::shared_ptr<block_loader> m_loader;
    const size_t                  m_block_count;
    bool                          m_cache_owner;
    int                           m_ownership_lock;
};
