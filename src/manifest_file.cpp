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

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>

#include "manifest_file.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"
#include "block.hpp"

using namespace std;
using namespace nervana;

const string manifest_file::m_file_type_id        = "FILE";
const string manifest_file::m_binary_type_id      = "BINARY";
const string manifest_file::m_string_type_id      = "STRING";
const string manifest_file::m_ascii_int_type_id   = "ASCII_INT";
const string manifest_file::m_ascii_float_type_id = "ASCII_FLOAT";

namespace errors
{
    const string no_header =
        "Metadata must be defined before any data. Potentially old manifest format used. Read "
        "neon doc/source/loading_data.rst#aeon-dataloader for more information.";
}

manifest_file::manifest_file(const string& filename,
                             bool          shuffle,
                             const string& root,
                             float         subset_fraction,
                             size_t        block_size,
                             uint32_t      seed,
                             uint32_t      node_id,
                             uint32_t      node_count,
                             int           batch_size)
    : m_source_filename(filename)
    , m_record_count{0}
    , m_node_id(node_id)
    , m_node_count(node_count)
    , m_block_size(block_size)
    , m_batch_size(batch_size)
    , m_shuffle{shuffle}
    , m_random{seed ? seed : random_device{}()}
{
    // for now parse the entire manifest on creation
    ifstream infile(m_source_filename);

    if (!infile.is_open())
    {
        throw std::runtime_error("Manifest file " + m_source_filename + " doesn't exist.");
    }

    initialize(infile, block_size, root, subset_fraction);
}

manifest_file::manifest_file(std::istream&      stream,
                             bool               shuffle,
                             const std::string& root,
                             float              subset_fraction,
                             size_t             block_size,
                             uint32_t           seed,
                             uint32_t           node_id,
                             uint32_t           node_count,
                             int                batch_size)
    : m_record_count{0}
    , m_node_id(node_id)
    , m_node_count(node_count)
    , m_block_size(block_size)
    , m_batch_size(batch_size)
    , m_shuffle{shuffle}
    , m_random{seed ? seed : random_device{}()}
{
    initialize(stream, block_size, root, subset_fraction);
}

string manifest_file::version()
{
    stringstream ss;
    ss << setfill('0') << setw(8) << hex << get_crc();
    return ss.str();
}

void manifest_file::initialize(std::istream&      stream,
                               size_t             block_size,
                               const std::string& root,
                               float              subset_fraction)
{
    // parse istream is and load the entire thing into m_record_list
    size_t                 element_count = 0;
    size_t                 line_number   = 0;
    string                 line;


    // read in each line, then from that istringstream, break into
    // tab-separated elements.
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
        }
        else if (line[0] == m_metadata_char)
        {
            // trim off the metadata char at the beginning of the line
            line = line.substr(1);
            if (m_element_types.empty() == false)
            {
                // Element types must be defined before any data
                throw std::invalid_argument(errors::no_header);
            }
            vector<string> element_list = split(line, m_delimiter_char);
            for (const string& type : element_list)
            {
                if (type == get_file_type_id())
                {
                    m_element_types.push_back(element_t::FILE);
                }
                else if (type == get_binary_type_id())
                {
                    m_element_types.push_back(element_t::BINARY);
                }
                else if (type == get_string_type_id())
                {
                    m_element_types.push_back(element_t::STRING);
                }
                else if (type == get_ascii_int_type_id())
                {
                    m_element_types.push_back(element_t::ASCII_INT);
                }
                else if (type == get_ascii_float_type_id())
                {
                    m_element_types.push_back(element_t::ASCII_FLOAT);
                }
                else
                {
                    ostringstream ss;
                    ss << "invalid metadata type '" << type;
                    ss << "' at line " << line_number;
                    throw std::invalid_argument(ss.str());
                }
            }
            element_count = m_element_types.size();
        }
        else if (line[0] == m_comment_char)
        {
            // Skip comments and empty lines
        }
        else
        {
            vector<string> element_list = split(line, m_delimiter_char);
            if (m_element_types.empty())
            {
                throw std::invalid_argument(errors::no_header);
            }

            if (element_list.size() != element_count)
            {
                ostringstream ss;
                ss << "at line: " << line_number;
                ss << ", manifest file has a line with differing number of elements (";
                ss << element_list.size() << ") vs (" << element_count << "): ";

                std::copy(element_list.begin(),
                          element_list.end(),
                          ostream_iterator<std::string>(ss, " "));
                throw std::runtime_error(ss.str());
            }
            m_record_list.push_back(element_list);
        }
        line_number++;
    }

    affirm(subset_fraction > 0.0 && subset_fraction <= 1.0,
           "subset_fraction must be >= 0 and <= 1");
    generate_subset(m_record_list, subset_fraction);

    m_record_count = m_record_list.size();

    // At this point the manifest is complete and ready to use
    // compute the crc now because we are going to add the manifest_root
    // to the records
    for (const vector<string>& record : m_record_list)
    {
        for (const string& s : record)
        {
            m_crc_engine.Update((const uint8_t*)s.data(), s.size());
        }
    }
    m_crc_engine.TruncatedFinal((uint8_t*)&m_computed_crc, sizeof(m_computed_crc));

    if (!root.empty())
    {
        for (size_t record_number = 0; record_number < m_record_list.size(); record_number++)
        {
            for (int i = 0; i < m_element_types.size(); i++)
            {
                if (m_element_types[i] == element_t::FILE)
                {
                    m_record_list[record_number][i] =
                        file_util::path_join(root, m_record_list[record_number][i]);
                }
            }
        }
    }

    generate_blocks();
}

const std::vector<manifest_file::element_t>& manifest_file::get_element_types() const
{
    return m_element_types;
}

vector<manifest_file::record_t>* manifest_file::next()
{
    vector<vector<string>>* rc = nullptr;
    if (m_counter < m_block_list.size())
    {
        auto load_index = m_block_load_sequence[m_counter];
        if (m_node_count == 0 )
        {
            rc  = &(m_block_list[load_index]);
        }
        else
        {
            // work around for multinode !!!!!!!!!!!!!
            // /////////////////////////////////////////////////////
            m_tmp_blocks.push_back(m_block_list[load_index]);
            if (m_tmp_blocks.size() > 4) m_tmp_blocks.pop_front();
            rc = &m_tmp_blocks.back();
            // //////////////////////////////////////////////////
        }

        m_counter++;
    }
    return rc;
}

void manifest_file::generate_blocks()
{
        if (m_shuffle)
            std::shuffle(m_record_list.begin(), m_record_list.end(), m_random);

        vector<record_t> record_list_shuffled;
        record_list_shuffled.resize(m_record_list.size());

        if (m_node_count != 0 )
        {
            m_record_count = m_record_list.size() / m_node_count;
            int batches = m_record_count / m_batch_size;
            record_list_shuffled.resize(m_record_count);

            for (int i = 0; i < batches * m_batch_size; i++)
            {
                auto batch_num = i / m_batch_size;
                auto index_in_batch = i % m_batch_size;
                record_list_shuffled[i] = m_record_list[batch_num * m_batch_size * m_node_count + m_batch_size * m_node_id + index_in_batch];
            }
            int tail_count = m_record_count - batches * m_batch_size;
            int tail_src   = batches * m_batch_size * m_node_count + tail_count * m_node_id;
            int tail_dst   = batches * m_batch_size;
            for (int i = 0; i < tail_count; i++)
                record_list_shuffled[i + tail_dst] = m_record_list[i + tail_src];
        }
        else
        {
            // TODO rewrite this in more performance friendly way
            record_list_shuffled = m_record_list;
        }

        // ///////////////////////////////////////////////////////////////////////////////////
        // reset block list

        std::vector<block_info> block_list = generate_block_list(m_record_count, m_block_size);
        m_block_list.clear();
        for (auto info : block_list)
        {
            vector<vector<string>> block;
            for (int i = info.start(); i < info.end(); i++)
            {
                block.push_back(record_list_shuffled[i]);
            }
            m_block_list.push_back(block);
        }

        m_block_load_sequence.reserve(m_block_list.size());
        m_block_load_sequence.resize(m_block_list.size());
        iota(m_block_load_sequence.begin(), m_block_load_sequence.end(), 0);
}

void manifest_file::reset()
{
    if (m_shuffle)
    {
        shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_random);
        if (m_node_count != 0 )
            generate_blocks();
    }
    m_counter = 0;
}

void manifest_file::generate_subset(vector<vector<string>>& record_list, float subset_fraction)
{
    if (subset_fraction < 1.0)
    {
        std::bernoulli_distribution distribution(subset_fraction);
        std::default_random_engine  generator(0); //get_global_random_seed());
        vector<record_t>              tmp;
        tmp.swap(record_list);
        size_t expected_count = tmp.size() * subset_fraction;
        size_t needed         = expected_count;

        for (int i = 0; i < tmp.size(); i++)
        {
            size_t remainder = tmp.size() - i;
            if ((needed == remainder) || distribution(generator))
            {
                record_list.push_back(tmp[i]);
                needed--;
                if (needed == 0)
                    break;
            }
        }
    }
}

uint32_t manifest_file::get_crc()
{
    return m_computed_crc;
}

const std::vector<std::string>& manifest_file::operator[](size_t offset) const
{
    for (const vector<record_t>& block : m_block_list)
    {
        if (offset < block.size())
        {
            return block[offset];
        }
        else
        {
            offset -= block.size();
        }
    }
    throw out_of_range("record not found in manifest");
}
