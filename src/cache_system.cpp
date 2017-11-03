/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <fstream>
#include <random>

#include "cache_system.hpp"
#include "file_util.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

const std::string cache_system::m_owner_lock_filename     = "caching_in_progress";
const std::string cache_system::m_cache_complete_filename = "cache_complete";
mutex             cache_system::m_mutex;

cache_system::cache_system(source_uid_t       uid,
                           size_t             block_count,
                           size_t             elements_per_record,
                           const std::string& cache_root,
                           bool               shuffle_enabled,
                           uint32_t           seed)
    : m_block_count(block_count)
    , m_cache_root(cache_root)
    , m_shuffle_enabled(shuffle_enabled)
    , m_elements_per_record(elements_per_record)
    , m_current_block_number{0}
    , m_random{seed ? seed : random_device{}()}
{
    m_block_load_sequence.resize(m_block_count);
    iota(m_block_load_sequence.begin(), m_block_load_sequence.end(), 0);

    stringstream ss;
    ss << "aeon_cache_";
    ss << hex << setw(8) << setfill('0') << uid;
    m_cache_dir = file_util::path_join(m_cache_root, ss.str());

    if (file_util::exists(m_cache_dir) == false)
    {
        file_util::make_directory(m_cache_dir);
    }
    try_get_access();
}

cache_system::~cache_system()
{
    if (is_ownership())
        release_ownership(m_cache_dir, m_cache_lock);
}

void cache_system::restart()
{
    m_current_block_number = 0;
}
void cache_system::try_get_access()
{
    m_current_block_number = 0;
    m_stage                = complete;
    lock_guard<mutex> lg(m_mutex);
    if (!check_if_complete(m_cache_dir))
        m_stage = take_ownership(m_cache_dir, m_cache_lock) ? ownership : blocked;
}

void cache_system::load_block(encoded_record_list& buffer)
{
    string block_name      = create_cache_block_name(m_block_load_sequence[m_current_block_number]);
    string block_file_path = file_util::path_join(m_cache_dir, block_name);

    if (file_util::exists(block_file_path))
    {
        ifstream file(block_file_path);
        if (file)
        {
            cpio::reader reader(file);

            if (reader.record_count() == 0)
                throw runtime_error("cache system: cache file corrupted");

            for (size_t record_number = 0; record_number < reader.record_count(); record_number++)
            {
                encoded_record record;
                for (size_t element = 0; element < m_elements_per_record; element++)
                {
                    vector<char> e;
                    reader.read(e);
                    record.add_element(std::move(e));
                }
                buffer.add_record(record);
            }
            if (m_shuffle_enabled)
                buffer.shuffle(std::random_device{}());
        }
    }
    else
        throw runtime_error("cache system: cache file missed");

    if (++m_current_block_number == m_block_count)
    {
        m_current_block_number = 0;
        if (m_shuffle_enabled)
            shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_random);
    }
}

void cache_system::store_block(const encoded_record_list& buffer)
{
    string block_name      = create_cache_block_name(m_current_block_number);
    string block_file_path = file_util::path_join(m_cache_dir, block_name);

    ofstream fi(block_file_path);
    if (fi)
    {
        cpio::writer writer(fi);
        writer.write_all_records(buffer);
    }
    else
        throw runtime_error("cache system: unable to write cache file");

    if (++m_current_block_number == m_block_count)
    {
        m_current_block_number = 0;
        // Done writing cache so unlock it
        {
            lock_guard<mutex> lg(m_mutex);
            mark_cache_complete(m_cache_dir);
            release_ownership(m_cache_dir, m_cache_lock);
        }
        m_stage = complete;
        if (m_shuffle_enabled)
            shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_random);
    }
}

string cache_system::create_cache_name(source_uid_t uid)
{
    stringstream ss;
    ss << "aeon_cache_";
    ss << hex << setw(8) << setfill('0') << uid;
    return ss.str();
}

string cache_system::create_cache_block_name(size_t block_number)
{
    stringstream ss;
    ss << "block_" << block_number << ".cpio";
    return ss.str();
}

bool cache_system::check_if_complete(const std::string& cache_dir)
{
    string file = file_util::path_join(cache_dir, m_cache_complete_filename);
    return file_util::exists(file);
}

void cache_system::mark_cache_complete(const std::string& cache_dir)
{
    string   file = file_util::path_join(cache_dir, m_cache_complete_filename);
    ofstream f{file};
}

bool cache_system::take_ownership(const std::string& cache_dir, int& lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    lock        = file_util::try_get_lock(file);
    return lock != -1;
}

void cache_system::release_ownership(const std::string& cache_dir, int lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    file_util::release_lock(lock, file);
}
