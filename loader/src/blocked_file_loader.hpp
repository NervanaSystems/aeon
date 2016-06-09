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

/* BlockedFIleLoader
 *
 * Loads blocks of files from a manifest.
 *
 * TODO: rename to BatchFileLoader since there is no temporal blocking
 * being done here, only a batch of file loads
 */
class BlockedFileLoader {
public:
    BlockedFileLoader(Manifest* manifest, uint object_size, uint target_size);

    void loadBlock(BufferPair& dest, uint i, uint block_size);
    void loadFile(char* dest, uint size, const string& filename);
    

private:
    void assert_exists_and_size(const string& filename, uint size);
    
    const Manifest* _manifest;
    uint _object_size;
    uint _target_size;
};
