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

#include "manifest.hpp"
#include "async_manager.hpp"
#include "buffer_batch.hpp"

namespace nervana
{
    class network_client;
    class manifest_nds_builder;
    class manifest_nds;
}

class nervana::network_client
{
public:
    network_client(const std::string& baseurl,
                   const std::string& token,
                   size_t             collection_id,
                   size_t             block_size,
                   size_t             shard_count,
                   size_t             shard_index);
    network_client(const network_client&) = default;

    ~network_client();

    static size_t callback(void* ptr, size_t size, size_t nmemb, void* stream);

    void get(const std::string& url, std::stringstream& stream);

    std::string load_block_url(size_t block_num);

    std::string metadata_url();

private:
    const std::string m_baseurl;
    const std::string m_token;
    const int         m_collection_id;
    const int         m_shard_count;
    const int         m_shard_index;
    unsigned int      m_object_count;
    unsigned int      m_block_count;
    uint32_t          m_macrobatch_size;
};

class nervana::manifest_nds_builder
{
public:
    manifest_nds_builder& filename(const std::string& filename);
    manifest_nds_builder& base_url(const std::string& url);
    manifest_nds_builder& token(const std::string& token);
    manifest_nds_builder& collection_id(size_t collection_id);
    manifest_nds_builder& block_size(size_t block_size);
    manifest_nds_builder& elements_per_record(size_t elements_per_record);
    manifest_nds_builder& shard_count(size_t shard_count);
    manifest_nds_builder& shard_index(size_t shard_index);
    manifest_nds_builder& shuffle(bool enable);
    manifest_nds_builder& seed(uint32_t seed);
    manifest_nds                  create();
    std::shared_ptr<manifest_nds> make_shared();

private:
    void parse_json(const std::string& filename);

    std::string m_base_url;
    std::string m_token;
    size_t      m_collection_id       = -1;
    size_t      m_block_size          = 5000;
    size_t      m_elements_per_record = -1;
    size_t      m_shard_count         = 1;
    size_t      m_shard_index         = 0;
    size_t      m_shuffle             = false;
    uint32_t    m_seed                = 0;
};

class nervana::manifest_nds : public nervana::async_manager_source<encoded_record_list>,
                              public nervana::manifest
{
    friend class manifest_nds_builder;

public:
    manifest_nds(const manifest_nds&) = default;
    virtual ~manifest_nds() {}
    encoded_record_list* next() override;
    void                 reset() override
    {
        if (m_shuffle)
        {
            shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_rnd);
        }
        m_current_block_number = 0;
    }

    size_t               record_count() const override { return m_record_count; }
    size_t               elements_per_record() const override { return 2; }
    size_t               block_count() const { return m_block_count; }
    encoded_record_list* load_block(size_t block_index);

    std::string cache_id() override;

    // NDS manifests doesn't have versions since collections are immutable
    std::string version() override { return ""; }
    static bool is_likely_json(const std::string filename);

private:
    void load_metadata();

    const std::string   m_base_url;
    const std::string   m_token;
    const size_t        m_collection_id;
    const size_t        m_block_size;
    const size_t        m_elements_per_record;
    const size_t        m_shard_count;
    const size_t        m_shard_index;
    size_t              m_record_count;
    size_t              m_block_count;
    network_client      m_network_client;
    size_t              m_current_block_number;
    std::vector<size_t> m_block_load_sequence;
    encoded_record_list m_current_block;
    bool                m_shuffle;
    std::minstd_rand0   m_rnd;

    manifest_nds() = delete;
    manifest_nds(const std::string& base_url,
                 const std::string& token,
                 size_t             collection_id,
                 size_t             block_size,
                 size_t             elements_per_record,
                 size_t             shard_count,
                 size_t             shard_index,
                 bool               shuffle,
                 uint32_t           seed = 0);

    static size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream);
};
