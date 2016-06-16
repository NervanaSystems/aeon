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
    BatchLoaderCPIOCache(const string& rootCacheDir,
                         const string& hash, const string& version,
                         shared_ptr<BatchLoader> loader);

    void loadBlock(BufferPair& dest, uint block_num, uint block_size);
    uint objectCount();

private:
    bool loadBlockFromCache(BufferPair& dest, uint block_num, uint block_size);
    void writeBlockToCache(BufferPair& dest, uint block_num, uint block_size);
    std::string blockFilename(uint block_num, uint block_size);

    void invalidateOldCache(const string& rootCacheDir, const string& hash, const string& version);
    bool filenameHoldsInvalidCache(const string& filename, const string& hash, const string& version);
    void removeDirectory(const string& dir);
    void makeDirectory(const string& dir);

    string _cacheDir;
    shared_ptr<BatchLoader> _loader;
};
