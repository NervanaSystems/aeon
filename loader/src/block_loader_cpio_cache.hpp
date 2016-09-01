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

namespace nervana {
    class block_loader_cpio_cache;
}

class nervana::block_loader_cpio_cache : public block_loader {
public:
    block_loader_cpio_cache(const std::string& rootCacheDir,
                            const std::string& cache_id, const std::string& version,
                            std::shared_ptr<block_loader> loader);

    void loadBlock(nervana::buffer_in_array& dest, uint32_t block_num);
    uint32_t objectCount();

private:
    bool loadBlockFromCache(nervana::buffer_in_array& dest, uint32_t block_num);
    void writeBlockToCache(nervana::buffer_in_array& dest, uint32_t block_num);
    std::string blockFilename(uint32_t block_num);

    void invalidateOldCache(const std::string& rootCacheDir, const std::string& cache_id, const std::string& version);
    bool filenameHoldsInvalidCache(const std::string& filename, const std::string& cache_id, const std::string& version);
    void removeDirectory(const std::string& dir);
    void makeDirectory(const std::string& dir);
    static int rm(const char *path, const struct stat *s, int flag, struct FTW *f);

    std::string _cacheDir;
    std::shared_ptr<block_loader> _loader;
};
