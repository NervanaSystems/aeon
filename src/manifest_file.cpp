/*
 Copyright 2016 Intel(R) Nervana(TM)
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
                             uint32_t      seed)
    : m_source_filename(filename)
    , m_record_count{0}
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
                             uint32_t           seed)
    : m_record_count{0}
    , m_shuffle{shuffle}
    , m_random{seed ? seed : random_device{}()}
{
    initialize(stream, block_size, root, subset_fraction);
}

string manifest_file::cache_id()
{
    // returns a hash of the m_filename
    std::size_t  h = std::hash<std::string>()(m_source_filename);
    stringstream ss;
    ss << setfill('0') << setw(16) << hex << h;
    return ss.str();
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
    vector<vector<string>> record_list;

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
            record_list.push_back(element_list);
        }
        line_number++;
    }

    affirm(subset_fraction > 0.0 && subset_fraction <= 1.0,
           "subset_fraction must be >= 0 and <= 1");
    generate_subset(record_list, subset_fraction);

    m_record_count = record_list.size();

    // At this point the manifest is complete and ready to use
    // compute the crc now because we are going to add the manifest_root
    // to the records
    for (const vector<string>& record : record_list)
    {
        for (const string& s : record)
        {
            m_crc_engine.Update((const uint8_t*)s.data(), s.size());
        }
    }
    m_crc_engine.TruncatedFinal((uint8_t*)&m_computed_crc, sizeof(m_computed_crc));

    if (m_shuffle)
        std::shuffle(record_list.begin(), record_list.end(), m_random);

    if (!root.empty())
    {
        for (size_t record_number = 0; record_number < record_list.size(); record_number++)
        {
            for (int i = 0; i < m_element_types.size(); i++)
            {
                if (m_element_types[i] == element_t::FILE)
                {
                    record_list[record_number][i] =
                        file_util::path_join(root, record_list[record_number][i]);
                }
            }
        }
    }

    // now that we have a list of all records, create blocks
    std::vector<block_info> block_list = generate_block_list(m_record_count, block_size);
    for (auto info : block_list)
    {
        vector<vector<string>> block;
        for (int i = info.start(); i < info.end(); i++)
        {
            block.push_back(record_list[i]);
        }
        m_block_list.push_back(block);
    }

    m_block_load_sequence.reserve(m_block_list.size());
    m_block_load_sequence.resize(m_block_list.size());
    iota(m_block_load_sequence.begin(), m_block_load_sequence.end(), 0);
}

const std::vector<manifest_file::element_t>& manifest_file::get_element_types() const
{
    return m_element_types;
}

vector<vector<string>>* manifest_file::next()
{
    vector<vector<string>>* rc = nullptr;
    if (m_counter < m_block_list.size())
    {
        auto load_index = m_block_load_sequence[m_counter];
        rc              = &(m_block_list[load_index]);
        m_counter++;
    }
    return rc;
}

void manifest_file::reset()
{
    if (m_shuffle)
    {
        shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_random);
    }
    m_counter = 0;
}

void manifest_file::generate_subset(vector<vector<string>>& record_list, float subset_fraction)
{
    if (subset_fraction < 1.0)
    {
        std::bernoulli_distribution distribution(subset_fraction);
        std::default_random_engine  generator(0); //get_global_random_seed());
        vector<record>              tmp;
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
    for (const vector<record>& block : m_block_list)
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
