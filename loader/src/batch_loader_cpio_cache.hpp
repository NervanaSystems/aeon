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

#include "batch_file_loader.hpp"

/* BatchLoaderCPIOCache
 *
 * caches loadBlock function in BatchLoader out to cpio files in
 * `cacheDir`.
 *
 * The hash is used to unquely identify a particular dataset and the version
 * is used to help invalidate old versions of the same dataset.  If a cache is
 * created with the same hash as an existing cache, but a different version,
 * old version is deleted.
 */

class BatchLoaderCPIOCache : public BatchLoader {
public:
    BatchLoaderCPIOCache(const std::string& rootCacheDir,
                         const std::string& hash, const std::string& version,
                         std::shared_ptr<BatchLoader> loader);

    void loadBlock(buffer_in_array& dest, uint block_num);
    uint objectCount();

private:
    bool loadBlockFromCache(buffer_in_array& dest, uint block_num);
    void writeBlockToCache(buffer_in_array& dest, uint block_num);
    std::string blockFilename(uint block_num);

    void invalidateOldCache(const std::string& rootCacheDir, const std::string& hash, const std::string& version);
    bool filenameHoldsInvalidCache(const std::string& filename, const std::string& hash, const std::string& version);
    void removeDirectory(const std::string& dir);
    void makeDirectory(const std::string& dir);
    static int rm(const char *path, const struct stat *s, int flag, struct FTW *f);

    std::string _cacheDir;
    std::shared_ptr<BatchLoader> _loader;
};
