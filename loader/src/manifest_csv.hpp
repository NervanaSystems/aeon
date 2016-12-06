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

#include <vector>
#include <string>
#include <random>

#include "manifest.hpp"
#include "async_manager.hpp"
#include "crc.hpp"

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
    class manifest_csv;
}

class nervana::manifest_csv : public nervana::async_manager_source<std::vector<std::string>>,
                              public nervana::manifest
{
public:
    manifest_csv(const std::string& filename,
                 bool shuffle,
                 const std::string& root = "",
                 float subset_fraction = 1.0);

    virtual ~manifest_csv() {}
    typedef std::vector<std::string>                  FilenameList;
    typedef std::vector<FilenameList>::const_iterator iter;

    std::string cache_id();
    std::string version();
    virtual size_t      object_count() override { return m_filename_lists.size(); }

    virtual std::vector<std::string>* next() override;

    virtual size_t      element_count() override
    {
        return m_filename_lists.size() > 0 ? m_filename_lists[0].size() : 0;
    }

    // begin and end provide iterators over the FilenameLists
    iter begin() const { return m_filename_lists.begin(); }
    iter end() const { return m_filename_lists.end(); }
    void generate_subset(float subset_fraction);
    uint32_t get_crc();

protected:
    void parse_stream(std::istream& is, const std::string& root);

private:
    const std::string         m_filename;
    std::vector<FilenameList> m_filename_lists;
    CryptoPP::CRC32C          m_crc_engine;
    bool                      m_crc_computed = false;
    uint32_t                  m_computed_crc;
    size_t                    m_counter{0};
};
