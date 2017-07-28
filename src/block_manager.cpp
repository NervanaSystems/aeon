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

#include <exception>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>

#include "block_manager.hpp"
#include "file_util.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

const std::string block_manager::m_owner_lock_filename     = "caching_in_progress";
const std::string block_manager::m_cache_complete_filename = "cache_complete";

nervana::block_manager::block_manager(block_loader_source* file_loader,
                                      size_t               block_size,
                                      const string&        cache_root,
                                      bool                 enable_shuffle)
    : async_manager<encoded_record_list, encoded_record_list>{file_loader, "block_manager"}
    , m_file_loader{*file_loader}
    , m_block_size{m_file_loader.block_size()}
    , m_block_count{m_file_loader.block_count()}
    , m_record_count{m_file_loader.record_count()}
    , m_current_block_number{0}
    , m_elements_per_record{m_file_loader.elements_per_record()}
    , m_cache_root{cache_root}
    , m_cache_enabled{m_cache_root.empty() == false}
    , m_shuffle_enabled{enable_shuffle}
    , m_block_load_sequence{}
    , m_rnd{get_global_random_seed()}
{
    m_block_load_sequence.reserve(m_block_count);
    m_block_load_sequence.resize(m_block_count);
    iota(m_block_load_sequence.begin(), m_block_load_sequence.end(), 0);
    if (m_cache_enabled)
    {
        m_source_uid = file_loader->get_uid();
        stringstream ss;
        ss << "aeon_cache_";
        ss << hex << setw(8) << setfill('0') << m_source_uid;
        m_cache_dir = file_util::path_join(m_cache_root, ss.str());

        if (file_util::exists(m_cache_dir) == false)
        {
            file_util::make_directory(m_cache_dir);
        }
        if (!check_if_complete(m_cache_dir))
        {
            if (!take_ownership(m_cache_dir, m_cache_lock))
            {
                throw runtime_error("dataset cache in process, try later");
            }
        }
    }
}

nervana::encoded_record_list* block_manager::filler()
{
    m_state                    = async_state::wait_for_buffer;
    encoded_record_list* rc    = get_pending_buffer();
    m_state                    = async_state::processing;
    encoded_record_list* input = nullptr;

    rc->clear();

    if (m_cache_enabled)
    {
        // cache path
        string block_name = create_cache_block_name(m_block_load_sequence[m_current_block_number]);
        string block_file_path = file_util::path_join(m_cache_dir, block_name);

        if (file_util::exists(block_file_path))
        {
            m_cache_hit++;
            ifstream f(block_file_path);
            if (f)
            {
                cpio::reader reader(f);

                if (reader.record_count() == 0)
                    throw runtime_error("block manager: cache file corrupted");

                for (size_t record_number = 0; record_number < reader.record_count();
                     record_number++)
                {
                    encoded_record record;
                    for (size_t element = 0; element < m_elements_per_record; element++)
                    {
                        vector<char> e;
                        reader.read(e);
                        record.add_element(std::move(e));
                    }
                    rc->add_record(record);
                }
                if (m_shuffle_enabled)
                {
                    std::random_device rd;
                    rc->shuffle(rd());
                }
            }
        }
        else
        {
            m_cache_miss++;
            m_state = async_state::fetching_data;
            input   = m_source->next();
            m_state = async_state::processing;
            if (input == nullptr)
            {
                rc = nullptr;
            }
            else
            {
                ofstream f(block_file_path);
                if (f)
                {
                    cpio::writer writer(f);
                    writer.write_all_records(*input);
                }
                input->swap(*rc);
            }
        }

        if (++m_current_block_number == m_block_count)
        {
            m_current_block_number = 0;

            // Done writing cache so unlock it
            mark_cache_complete(m_cache_dir);
            release_ownership(m_cache_dir, m_cache_lock);
        }
    }
    else
    {
        // The non-cache path
        m_state = async_state::fetching_data;
        input   = m_source->next();
        m_state = async_state::processing;

        if (input != nullptr)
        {
            rc->swap(*input);

            if (++m_current_block_number == m_block_count)
            {
                m_current_block_number = 0;
                m_file_loader.reset();
            }
        }
    }

    if (m_shuffle_enabled && m_current_block_number == 0)
    {
        // This will not trigger on the first pass through the dataset
        shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_rnd);
    }

    if (rc && rc->size() == 0)
    {
        rc = nullptr;
    }

    m_state = async_state::idle;
    return rc;
}

string block_manager::create_cache_name(source_uid_t uid)
{
    stringstream ss;
    ss << "aeon_cache_";
    ss << hex << setw(8) << setfill('0') << uid;
    return ss.str();
}

string block_manager::create_cache_block_name(size_t block_number)
{
    stringstream ss;
    ss << "block_" << block_number << ".cpio";
    return ss.str();
}

bool block_manager::check_if_complete(const std::string& cache_dir)
{
    string file = file_util::path_join(cache_dir, m_cache_complete_filename);
    return file_util::exists(file);
}

void block_manager::mark_cache_complete(const std::string& cache_dir)
{
    string   file = file_util::path_join(cache_dir, m_cache_complete_filename);
    ofstream f{file};
}

bool block_manager::take_ownership(const std::string& cache_dir, int& lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    lock        = file_util::try_get_lock(file);
    return lock != -1;
}

void block_manager::release_ownership(const std::string& cache_dir, int lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    file_util::release_lock(lock, file);
}
