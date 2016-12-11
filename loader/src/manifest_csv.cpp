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

#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include "manifest_csv.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

manifest_csv::manifest_csv(const string& filename, bool shuffle, const string& root,
                           float subset_fraction)
    : m_source_filename(filename)
{
    // for now parse the entire manifest on creation
    ifstream infile(m_source_filename);

    if (!infile.is_open())
    {
        throw std::runtime_error("Manifest file " + m_source_filename + " doesn't exist.");
    }

    initialize(infile, shuffle, root, subset_fraction);
}

manifest_csv::manifest_csv(std::istream& stream, bool shuffle, const std::string& root,
                           float subset_fraction)
{
    initialize(stream, shuffle, root, subset_fraction);
}

void manifest_csv::initialize(std::istream& stream, bool shuffle, const std::string& root,
                              float subset_fraction)
{
    parse_stream(stream, root);

    // If we don't need to shuffle, there may be small performance
    // benefits in some situations to stream the filename_lists instead
    // of loading them all at once.  That said, in the event that there
    // is no cache and we are resuming training at a specific epoch, we
    // may need to be able to jump around and read random blocks of the
    // file, so a purely stream based interface is not sufficient.
    if (shuffle)
    {
        std::shuffle(m_record_list.begin(), m_record_list.end(), std::mt19937(0));
    }

    affirm(subset_fraction > 0.0 && subset_fraction <= 1.0,
           "subset_fraction must be >= 0 and <= 1");
    generate_subset(subset_fraction);
}

string manifest_csv::cache_id()
{
    // returns a hash of the m_filename
    std::size_t  h = std::hash<std::string>()(m_source_filename);
    stringstream ss;
    ss << setfill('0') << setw(16) << hex << h;
    return ss.str();
}

string manifest_csv::version()
{
    stringstream ss;
    ss << setfill('0') << setw(8) << hex << get_crc();
    return ss.str();
}

void manifest_csv::parse_stream(istream& is, const string& root)
{
    // parse istream is and load the entire thing into m_record_list
    size_t previous_element_count = 0;
    size_t line_number = 0;
    string line;

    // read in each line, then from that istringstream, break into
    // comma-separated elements.
    while (std::getline(is, line))
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
                // Already have element types so this is an error
                // Element types must be defined before any data
                ostringstream ss;
                ss << "metadata must be defined before any data at line " << line_number;
                throw std::invalid_argument(ss.str());
            }
            vector<string> element_list = split(line, m_delimiter_char);
            for (const string& type : element_list)
            {
                if (type == "FILE")
                {
                    m_element_types.push_back(element_t::FILE);
                }
                else if (type == "BINARY")
                {
                    m_element_types.push_back(element_t::BINARY);
                }
                else if (type == "STRING")
                {
                    m_element_types.push_back(element_t::STRING);
                }
                else if (type == "ASCII_INT")
                {
                    m_element_types.push_back(element_t::ASCII_INT);
                }
                else if (type == "ASCII_FLOAT")
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
                // No element type metadata found so create defaults
                for (int i = 0; i < element_list.size(); i++)
                {
                    m_element_types.push_back(element_t::FILE);
                }
            }

            if (!root.empty())
            {
                for (int i = 0; i < element_list.size(); i++)
                {
                    if (m_element_types[i] == element_t::FILE)
                    {
                        element_list[i] = file_util::path_join(root, element_list[i]);
                    }
                }
            }

            if (line_number == 0)
            {
                previous_element_count = element_list.size();
            }

            if (element_list.size() != previous_element_count)
            {
                ostringstream ss;
                ss << "at line: " << line_number;
                ss << ", manifest file has a line with differing number of files (";
                ss << element_list.size() << ") vs (" << previous_element_count << "): ";

                std::copy(element_list.begin(), element_list.end(),
                          ostream_iterator<std::string>(ss, " "));
                throw std::runtime_error(ss.str());
            }
            previous_element_count = element_list.size();
            m_record_list.push_back(element_list);
            line_number++;
        }
    }
}

const std::vector<manifest_csv::element_t>& manifest_csv::get_element_types() const
{
    return m_element_types;
}

vector<string>* manifest_csv::next()
{
    vector<string>* rc = nullptr;
    if (m_counter < record_count())
    {
        rc = &(m_record_list[m_counter]);
        m_counter++;
    }
    return rc;
}

void manifest_csv::reset()
{
    m_counter = 0;
}

void manifest_csv::generate_subset(float subset_fraction)
{
    if (subset_fraction < 1.0)
    {
        m_crc_computed = false;
        std::bernoulli_distribution distribution(subset_fraction);
        std::default_random_engine  generator(get_global_random_seed());
        vector<record>              tmp;
        tmp.swap(m_record_list);
        size_t expected_count = tmp.size() * subset_fraction;
        size_t needed = expected_count;

        for (int i = 0; i < tmp.size(); i++)
        {
            size_t remainder = tmp.size() - i;
            if ((needed == remainder) || distribution(generator))
            {
                m_record_list.push_back(tmp[i]);
                needed--;
                if (needed == 0)
                    break;
            }
        }
    }
}

uint32_t manifest_csv::get_crc()
{
    if (m_crc_computed == false)
    {
        for (const vector<string>& tmp : m_record_list)
        {
            for (const string& s : tmp)
            {
                m_crc_engine.Update((const uint8_t*)s.data(), s.size());
            }
        }
        m_crc_engine.TruncatedFinal((uint8_t*)&m_computed_crc, sizeof(m_computed_crc));
        m_crc_computed = true;
    }
    return m_computed_crc;
}
