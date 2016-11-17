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

using namespace std;
using namespace nervana;

manifest_csv::manifest_csv(const string& filename, bool shuffle, const string& root, float subset_fraction)
    : m_filename(filename)
{
    // for now parse the entire manifest on creation
    ifstream infile(m_filename);

    if (!infile.is_open())
    {
        throw std::runtime_error("Manifest file " + m_filename + " doesn't exist.");
    }

    parse_stream(infile, root);

    // If we don't need to shuffle, there may be small performance
    // benefits in some situations to stream the filename_lists instead
    // of loading them all at once.  That said, in the event that there
    // is no cache and we are resuming training at a specific epoch, we
    // may need to be able to jump around and read random blocks of the
    // file, so a purely stream based interface is not sufficient.
    if (shuffle)
    {
        shuffle_filename_lists();
    }
}

string manifest_csv::cache_id()
{
    // returns a hash of the m_filename
    std::size_t  h = std::hash<std::string>()(m_filename);
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
    // parse istream is and load the entire thing into m_filename_lists
    uint32_t prev_num_fields = 0, lineno = 0;
    string   line;

    // read in each line, then from that istringstream, break into
    // comma-separated fields.
    while (std::getline(is, line))
    {
        if (line.empty() || line[0] == '#') // Skip comments and empty lines
        {
            continue;
        }

        vector<string> field_list = split(line, ',');
        if (!root.empty())
        {
            for (int i = 0; i < field_list.size(); i++)
            {
                field_list[i] = file_util::path_join(root, field_list[i]);
            }
        }

        if (lineno == 0)
        {
            prev_num_fields = field_list.size();
        }

        if (field_list.size() != prev_num_fields)
        {
            ostringstream ss;
            ss << "at line: " << lineno;
            ss << ", manifest file has a line with differing number of files (";
            ss << field_list.size() << ") vs (" << prev_num_fields << "): ";

            std::copy(field_list.begin(), field_list.end(), ostream_iterator<std::string>(ss, " "));
            throw std::runtime_error(ss.str());
        }
        prev_num_fields = field_list.size();
        m_filename_lists.push_back(field_list);
        lineno++;
    }
}

void manifest_csv::shuffle_filename_lists()
{
    // shuffles m_filename_lists.  It is possible that the order of the
    // filenames in the manifest file were in some sorted order and we
    // don't want our blocks to be biased by that order.

    // hardcode random seed to 0 since this step can be cached into a
    // CPIO file.  We don't want to cache anything that is based on a
    // changing random seed, so don't use a changing random seed.
    std::shuffle(m_filename_lists.begin(), m_filename_lists.end(), std::mt19937(0));
}

void manifest_csv::generate_subset(float subset_fraction)
{
    if (subset_fraction < 1.0)
    {
        m_crc_computed = false;
        std::bernoulli_distribution distribution(subset_fraction);
        std::default_random_engine  generator(get_global_random_seed());
        vector<FilenameList>        tmp;
        tmp.swap(m_filename_lists);
        size_t expected_count = tmp.size() * subset_fraction;
        size_t needed         = expected_count;

        for (int i = 0; i < tmp.size(); i++)
        {
            size_t remainder = tmp.size() - i;
            if ((needed == remainder) || distribution(generator))
            {
                m_filename_lists.push_back(tmp[i]);
                needed--;
                if (needed == 0)
                    break;
            }
        }
        //        cout << __FILE__ << " " << __LINE__ << " expected=" << expected_count << ", actual=" << m_filename_lists.size() <<
        //        endl;
    }
}

uint32_t manifest_csv::get_crc()
{
    if (m_crc_computed == false)
    {
        for (const vector<string>& tmp : m_filename_lists)
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

int manifest_csv::nelements()
{
    return m_filename_lists.size() > 0 ? m_filename_lists[0].size() : 0;
}
