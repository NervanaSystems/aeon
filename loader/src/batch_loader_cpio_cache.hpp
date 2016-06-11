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

using namespace std;

/* BatchLoaderCPIOCache
 *
 * caches loadBlock function in BatchLoader out to cpio files in
 * `cacheDir`.
 *
 */

class BatchLoaderCPIOCache : public BatchLoader {
public:
    BatchLoaderCPIOCache(const char* cacheDir, BatchLoader* loader);

    void loadBlock(BufferPair& dest, uint block_num, uint block_size);

private:
    bool loadBlockFromCache(BufferPair& dest, uint block_num, uint block_size);
    void writeBlockToCache(BufferPair& dest, uint block_num, uint block_size);
    string blockFilename(uint block_num, uint block_size);

    string _cacheDir;
    BatchLoader* _loader;
};
