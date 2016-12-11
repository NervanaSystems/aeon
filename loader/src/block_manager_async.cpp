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

#include <iomanip>
#include <fstream>
#include <exception>

#include "block_manager_async.hpp"
#include "file_util.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

const std::string block_manager_async::m_owner_lock_filename     = "caching_in_progress";
const std::string block_manager_async::m_cache_complete_filename = "cache_complete";

nervana::block_manager_async::block_manager_async(block_loader_source_async* file_loader, size_t block_size, const string& cache_root, bool enable_shuffle)
    : async_manager<variable_buffer_array, variable_buffer_array>{file_loader}
    , m_file_loader{*file_loader}
    , m_block_size{m_file_loader.block_size()}
    , m_block_count{m_file_loader.block_count()}
    , m_current_block_number{0}
    , m_elements_per_record{m_file_loader.element_count()}
    , m_cache_root{cache_root}
    , m_cache_enabled{m_cache_root.empty() == false}
    , m_shuffle_enabled{enable_shuffle}
{
    for (int k = 0; k < 2; ++k)
    {
        for (size_t j = 0; j < m_elements_per_record; ++j)
        {
            m_containers[k].emplace_back();
        }
    }

    if (m_cache_enabled)
    {
        m_source_uid = file_loader->get_uid();
        stringstream ss;
        ss << "aeon_cache_";
        ss << hex << setw(8) << setfill('0') << m_source_uid;
        m_cache_dir = file_util::path_join(m_cache_root, ss.str());

        if (file_util::exists(m_cache_dir))
        {
            if (!check_if_complete(m_cache_dir))
            {
                throw runtime_error("dataset cache in process, try later");
            }
        }
        else
        {
            file_util::make_directory(m_cache_dir);
            if (!take_ownership(m_cache_dir, m_cache_lock))
            {
                throw runtime_error("dataset cache error taking ownership");
            }
        }
    }
}

nervana::variable_buffer_array* block_manager_async::filler()
{
    variable_buffer_array* rc = get_pending_buffer();
    variable_buffer_array* input = nullptr;

    for (size_t i = 0; i < m_elements_per_record; ++i)
    {
        // These should be empty at this point
        rc->at(i).reset();
    }

    if (m_cache_enabled)
    {
        // cache path
        string block_name = create_cache_block_name(m_current_block_number);
        string block_file_path = file_util::path_join(m_cache_dir, block_name);

        if (file_util::exists(block_file_path))
        {
            m_cache_hit++;
            ifstream f(block_file_path);
            if (f)
            {
                cpio::reader reader(f);
                for (size_t record=0; record<reader.record_count(); record++)
                {
                    for (size_t element=0; element<m_elements_per_record; element++)
                    {
                        reader.read(rc->at(element));
                    }
                }
            }
        }
        else
        {
            m_cache_miss++;
            input = m_source->next();
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
                for (size_t i = 0; i < m_elements_per_record; ++i)
                {
                    rc->at(i) = input->at(i);
                }
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
        if (m_current_block_number == m_block_count)
        {
            m_current_block_number = 0;
            m_file_loader.reset();
        }

        input = m_source->next();

        m_current_block_number++;

        for (size_t i = 0; i < m_elements_per_record; ++i)
        {
            rc->at(i) = input->at(i);
        }
    }

    if (rc && rc->at(0).size() == 0)
    {
        rc = nullptr;
    }

    return rc;
}

void block_manager_async::move_src_to_dst(variable_buffer_array* src_array_ptr, variable_buffer_array* dst_array_ptr, size_t count)
{
    for (size_t ridx = 0; ridx < m_elements_per_record; ++ridx)
    {
        buffer_variable_size_elements& src = src_array_ptr->at(ridx);
        buffer_variable_size_elements& dst = dst_array_ptr->at(ridx);

        auto start_iter = src.begin();
        auto end_iter   = src.begin() + count;

        dst.append(make_move_iterator(start_iter), make_move_iterator(end_iter));
        src.erase(start_iter, end_iter);
    }
}

string block_manager_async::create_cache_name(source_uid_t uid)
{
    stringstream ss;
    ss << "aeon_cache_";
    ss << hex << setw(8) << setfill('0') << uid;
    return ss.str();
}

string block_manager_async::create_cache_block_name(size_t block_number)
{
    stringstream ss;
    ss << "block_" << block_number << ".cpio";
    return ss.str();
}

bool block_manager_async::check_if_complete(const std::string& cache_dir)
{
    string file = file_util::path_join(cache_dir, m_cache_complete_filename);
    return file_util::exists(file);
}

void block_manager_async::mark_cache_complete(const std::string& cache_dir)
{
    string   file = file_util::path_join(cache_dir, m_cache_complete_filename);
    ofstream f{file};
}

bool block_manager_async::take_ownership(const std::string& cache_dir, int& lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    lock = file_util::try_get_lock(file);
    return lock != -1;
}

void block_manager_async::release_ownership(const std::string& cache_dir, int lock)
{
    string file = file_util::path_join(cache_dir, m_owner_lock_filename);
    file_util::release_lock(lock, file);
}
