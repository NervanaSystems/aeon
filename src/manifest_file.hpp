/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <random>

#include "manifest.hpp"
#include "async_manager.hpp"
#include "crc.hpp"
#include "util.hpp"

/* Manifest
 *
 * load a manifest file and parse the filenames
 *
 * The format of the file should be something like:
 *
 * object_filename1,target_filename1
 * object_filename2,target_filename2
 * ...
 *
 * string hash() is to be used as a key for the cache.  It is possible
 * that it will be better to use the filename and last modified time as
 * a key instead.
 *
 */
namespace nervana
{
    class manifest_file;
}

class nervana::manifest_file
    : public nervana::async_manager_source<std::vector<std::vector<std::string>>>,
      public nervana::manifest
{
public:
    manifest_file(const std::string& filename,
                  bool               shuffle,
                  const std::string& root            = "",
                  float              subset_fraction = 1.0,
                  size_t             block_size      = 5000,
                  uint32_t           seed            = 0,
                  uint32_t           node_id         = 0,
                  uint32_t           node_count      = 0,
                  int                batch_size      = 1,
                  bool               drop_incomplete_batch = false);

    manifest_file(std::istream&      stream,
                  bool               shuffle,
                  const std::string& root            = "",
                  float              subset_fraction = 1.0,
                  size_t             block_size      = 5000,
                  uint32_t           seed            = 0,
                  uint32_t           node_id         = 0,
                  uint32_t           node_count      = 0,
                  int                batch_size      = 1,
                  bool               drop_incomplete_batch = false);

    virtual ~manifest_file() {}
    using record_t = std::vector<std::string>;

    std::string cache_id() override;
    std::string version() override;

    std::vector<record_t>* next() override;
    void                                   reset() override;

    size_t   block_count() const { return m_block_list.size(); }
    size_t   record_count() const override { return m_record_count; }
    size_t   elements_per_record() const override { return m_element_types.size(); }
    uint32_t get_crc();

    void generate_blocks();

    static char                   get_delimiter() { return m_delimiter_char; }
    static char                   get_comment_char() { return m_comment_char; }
    static char                   get_metadata_char() { return m_metadata_char; }
    static const std::string&     get_file_type_id() { return m_file_type_id; }
    static const std::string&     get_binary_type_id() { return m_binary_type_id; }
    static const std::string&     get_string_type_id() { return m_string_type_id; }
    static const std::string&     get_ascii_int_type_id() { return m_ascii_int_type_id; }
    static const std::string&     get_ascii_float_type_id() { return m_ascii_float_type_id; }
    const std::vector<element_t>& get_element_types() const;

    const std::vector<std::string>& operator[](size_t offset) const;

protected:
    void initialize(std::istream&      stream,
                    size_t             block_size,
                    const std::string& root,
                    float              subset_fraction);

private:
    void generate_subset(std::vector<std::vector<std::string>>&, float subset_fraction);

    std::vector<std::vector<std::string>>           m_record_list;
    std::string                      m_source_filename;
    std::vector<std::vector<record_t>> m_block_list;
    std::deque<std::vector<record_t>> m_tmp_blocks;
    CryptoPP::CRC32C                 m_crc_engine;
    uint32_t                         m_computed_crc;
    size_t                           m_counter{0};
    size_t                           m_record_count;
    uint32_t                         m_node_id{0};
    uint32_t                         m_node_count{0};
    size_t                           m_block_size;
    int                              m_batch_size{1};
    static const char                m_delimiter_char = '\t';
    static const char                m_comment_char   = '#';
    static const char                m_metadata_char  = '@';
    std::vector<element_t>           m_element_types;
    std::vector<size_t>              m_block_load_sequence;
    bool                             m_shuffle;
    bool                             m_drop_incomplete_batch;
    random_engine_t                  m_random;
    static const std::string         m_file_type_id;
    static const std::string         m_binary_type_id;
    static const std::string         m_string_type_id;
    static const std::string         m_ascii_int_type_id;
    static const std::string         m_ascii_float_type_id;
};
