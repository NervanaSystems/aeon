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

#include <sstream>
#include <string>

#include "buffer.hpp"
#include "cpio.hpp"
#include "batch_loader.hpp"

class NDSBatchLoader : public BatchLoader {
public:
    NDSBatchLoader(const std::string baseurl, int tag_id, int shard_count=1, int shard_index=0);
    ~NDSBatchLoader();

    void loadBlock(BufferPair& dest, uint block_num, uint block_size);
    uint objectCount();

    uint blockCount(uint block_size);

private:
    void get(const std::string url, std::stringstream& stream);

    const std::string url(uint block_num, uint block_size);
    const std::string _baseurl;
    const int _tag_id;
    const int _shard_count;
    const int _shard_index;

    // reuse connection across requests
    void* _curl;
};
