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
 */

class BatchLoaderCPIOCache : public BatchLoader {
public:
    BatchLoaderCPIOCache(const std::string& rootCacheDir,
                         const std::string& hash, const std::string& version,
                         std::shared_ptr<BatchLoader> loader);

    void loadBlock(BufferArray& dest, uint block_num, uint block_size);
    uint objectCount();

private:
    bool loadBlockFromCache(BufferArray& dest, uint block_num, uint block_size);
    void writeBlockToCache(BufferArray& dest, uint block_num, uint block_size);
    std::string blockFilename(uint block_num, uint block_size);

    void invalidateOldCache(const std::string& rootCacheDir, const std::string& hash, const std::string& version);
    bool filenameHoldsInvalidCache(const std::string& filename, const std::string& hash, const std::string& version);
    void removeDirectory(const std::string& dir);
    void makeDirectory(const std::string& dir);

    std::string _cacheDir;
    std::shared_ptr<BatchLoader> _loader;
};
