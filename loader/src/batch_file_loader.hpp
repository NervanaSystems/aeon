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

#include "manifest.hpp"
#include "buffer.hpp"
#include "batch_loader.hpp"

/* BatchFileLoader
 *
 * Loads blocks of files from a manifest into a Buffer.
 *
 */
class BatchFileLoader : public BatchLoader {
public:
    BatchFileLoader(shared_ptr<Manifest> manifest);

    void loadBlock(BufferPair& dest, uint block_num, uint block_size);
    void loadFile(Buffer* buff, const string& filename);
    uint objectCount();

private:
    off_t getFileSize(const string& filename);
    
    const shared_ptr<Manifest> _manifest;
};
