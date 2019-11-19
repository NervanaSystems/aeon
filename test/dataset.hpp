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

#include <sys/stat.h>
#include <string>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <fstream>

#include "file_util.hpp"

template <typename T>
class dataset
{
public:
    dataset()
        : m_path()
        , m_prefix("archive-")
        , m_max_items(4000)
        , m_max_size(-1)
        , m_set_size(100000)
        , m_path_existed(false)
    {
    }

    virtual ~dataset() {}
    T& directory(const std::string& dir)
    {
        m_path = dir;
        return *(T*)this;
    }

    T& prefix(const std::string& prefix)
    {
        m_prefix = prefix;
        return *(T*)this;
    }

    T& macrobatch_max_records(int max)
    {
        assert(max > 0);
        m_max_items = max;
        return *(T*)this;
    }

    T& macrobatch_max_size(int max)
    {
        assert(max > 0);
        m_max_size = max;
        return *(T*)this;
    }

    T& dataset_size(int size)
    {
        assert(size > 0);
        m_set_size = size;
        return *(T*)this;
    }

    void delete_files()
    {
        for (const std::string& f : m_file_list)
        {
            remove(f.c_str());
        }
        if (!m_path_existed)
        {
            // delete directory
            remove(m_path.c_str());
        }
    }

    std::string              get_dataset_path() { return m_path; }
    std::vector<std::string> get_files() { return m_file_list; }
protected:
    virtual std::vector<unsigned char> render_target(int datumNumber) = 0;
    virtual std::vector<unsigned char> render_datum(int datumNumber)  = 0;

private:
    bool exists(const std::string& fileName)
    {
        struct stat stats;
        return stat(fileName.c_str(), &stats) == 0;
    }

    std::string m_path;
    std::string m_prefix;
    int         m_max_items;
    int         m_max_size;
    int         m_set_size;
    bool        m_path_existed;

    std::vector<std::string> m_file_list;
};
