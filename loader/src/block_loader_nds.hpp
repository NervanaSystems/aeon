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

#include "buffer_in.hpp"
#include "cpio.hpp"
#include "block_loader.hpp"
#include "util.hpp"

namespace nervana
{
    class block_loader_nds;
}

class nervana::block_loader_nds : public block_loader
{
public:
    block_loader_nds(
            const std::string& baseurl,
            const std::string& token,
            int collection_id,
            uint32_t block_size,
            int shard_count=1,
            int shard_index=0
            );

    ~block_loader_nds();

    void load_block(nervana::buffer_in_array& dest, uint32_t block_num) override;
    void prefetch_block(uint32_t block_num) override;
    uint32_t object_count() override;

    uint32_t block_count();

private:
    void load_metadata();

    void get(const std::string& url, std::stringstream& stream);

    const std::string load_block_url(uint32_t block_num);
    const std::string metadata_url();
    void prefetch_entry(void* param);
    void fetch_block(uint32_t block_num);

    const std::string m_baseurl;
    const std::string m_token;
    const int         m_collection_id;
    const int         m_shard_count;
    const int         m_shard_index;
    unsigned int      m_object_count;
    unsigned int      m_block_count;

    // reuse connection across requests
    void*             m_curl;

    async                          m_async_handler;
    std::vector<std::vector<char>> m_prefetch_buffer;
    uint32_t                       m_prefetch_block_num;
    bool                           m_prefetch_pending = false;
};
