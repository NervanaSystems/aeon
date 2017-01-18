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

#include <errno.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>

#include "cpio.hpp"
#include "block_loader_cpio_cache.hpp"
#include "file_util.hpp"

using namespace std;
using namespace nervana;

block_loader_cpio_cache::block_loader_cpio_cache(const string& rootCacheDir, const string& cache_id, const string& version,
                                                 shared_ptr<block_loader> loader)
    : block_loader(loader->block_size())
    , m_loader(loader)
    , m_block_count{loader->block_count()}
    , m_cache_owner{false}
{
    invalidate_old_cache(rootCacheDir, cache_id, version);

    m_cache_dir = file_util::path_join(rootCacheDir, cache_id + "_" + version);

    if (file_util::make_directory(m_cache_dir))
    {
        // If I successfully created the directory then it did not exist.
        // Therefore I am the owner and must write the end-of-data file
        m_cache_owner = true;
    }

    if (check_if_complete() == false)
    {
        if (take_ownership() == false)
        {
            throw std::runtime_error("dataloader cache incomplete, try again later");
        }
    }
}

void block_loader_cpio_cache::load_block(buffer_in_array& dest, uint32_t block_num)
{
    if (load_block_from_cache(dest, block_num))
    {
        return;
    }
    else
    {
        m_loader->load_block(dest, block_num);

        try
        {
            write_block_to_cache(dest, block_num);

            if (block_num == m_block_count - 1)
            {
                mark_cache_complete();
                release_ownership();
            }
        }
        catch (std::exception& e)
        {
            // failure to write block to cache doesn't stop execution, only print an error
            cerr << "ERROR writing block to cache: " << e.what() << endl;
        }
    }
}

bool block_loader_cpio_cache::load_block_from_cache(buffer_in_array& dest, uint32_t block_num)
{
    // load a block from cpio cache into dest.  If file doesn't exist, return false.
    //  If loading from cpio cache was successful return true.
    bool rc = false;

    ifstream f(block_filename(block_num));
    if (f)
    {
        cpio::reader reader(f);
        // load cpio file into dest one record at a time
        for (int i = 0; i < reader.record_count(); ++i)
        {
            for (auto d : dest)
            {
                try
                {
                    reader.read(*d);
                }
                catch (std::exception& e)
                {
                    d->add_exception(std::current_exception());
                }
            }
        }
        reader.close();
        rc = true;
    }

    // cpio file was read successfully, no need to hit primary data
    // source
    return rc;
}

void block_loader_cpio_cache::write_block_to_cache(buffer_in_array& buff, uint32_t block_num)
{
    ofstream f{block_filename(block_num)};
    if (f)
    {
        cpio::writer writer(f);
        writer.write_all_records(buff);
    }
}

void block_loader_cpio_cache::invalidate_old_cache(const string& rootCacheDir, const string& cache_id, const string& version)
{
    // remove cache directories that match rootCacheDir and cache_id but not version

    DIR*           dir;
    struct dirent* ent;
    if ((dir = opendir(rootCacheDir.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            if (filename_holds_invalid_cache(ent->d_name, cache_id, version))
            {
                file_util::remove_directory(file_util::path_join(rootCacheDir, ent->d_name));
            }
        }
        closedir(dir);
    }
    else
    {
        throw std::runtime_error("error enumerating old cache in " + rootCacheDir);
    }
}

bool block_loader_cpio_cache::filename_holds_invalid_cache(const string& filename, const string& cache_id, const string& version)
{
    // in order for `filename` to hold invalid cache, it must begin with
    // `cache_id`, but not contain `version`

    if (filename.find(cache_id) != 0)
    {
        // filename doesn't start with cache_id, dont remove it
        return false;
    }
    if (filename.find(version) == string::npos)
    {
        // filename does start with cache_id, but doesnt have version, invalidate
        return true;
    }
    // filename does start with cache_id and does have version, keep, its valid
    return false;
}

string block_loader_cpio_cache::block_filename(uint32_t block_num)
{
    string file = to_string(block_num) + "-" + to_string(m_block_size) + ".cpio";
    string rc   = file_util::path_join(m_cache_dir, file);
    return rc;
}

uint32_t block_loader_cpio_cache::record_count()
{
    return m_loader->record_count();
}

string block_loader_cpio_cache::get_cache_dir() const
{
    return m_cache_dir;
}

bool block_loader_cpio_cache::check_if_complete()
{
    string file = file_util::path_join(m_cache_dir, m_cache_complete_filename);
    return file_util::exists(file);
}

void block_loader_cpio_cache::mark_cache_complete()
{
    string   file = file_util::path_join(m_cache_dir, m_cache_complete_filename);
    ofstream f{file};
}

bool block_loader_cpio_cache::take_ownership()
{
    string file      = file_util::path_join(m_cache_dir, m_owner_lock_filename);
    m_ownership_lock = file_util::try_get_lock(file);
    return m_ownership_lock != -1;
}

void block_loader_cpio_cache::release_ownership()
{
    string file = file_util::path_join(m_cache_dir, m_owner_lock_filename);
    file_util::release_lock(m_ownership_lock, file);
}

void block_loader_cpio_cache::prefetch_block(uint32_t block_num)
{
    string file = block_filename(block_num);
    if (file_util::exists(file) == false)
    {
        m_loader->prefetch_block(block_num);
    }
}
