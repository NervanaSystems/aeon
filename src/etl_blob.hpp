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

#pragma once

#include <sstream>

#include "interface.hpp"
#include "util.hpp"

namespace nervana
{
    namespace blob
    {
        class config;
        class decoded;

        class extractor;
        class loader;
    }
}

class nervana::blob::config : public interface::config
{
public:
    std::string output_type{"float"};
    size_t      output_count;
    std::string name;

    config(nlohmann::json js)
    {
        for (auto& info : config_list)
        {
            info->parse(js);
        }
        verify_config("blob", config_list, js);

        add_shape_type({output_count}, output_type);
    }

private:
    config() {}
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(output_count, mode::REQUIRED),
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v) {
            return output_type::is_valid_type(v);
        })};
};

class nervana::blob::decoded : public interface::decoded_media
{
    friend class loader;

public:
    decoded(const void* buf, size_t bufSize)
        : m_data{buf}
        , m_data_size{bufSize}
    {
    }

    virtual ~decoded() override {}
private:
    const void* m_data;
    size_t      m_data_size;
};

class nervana::blob::extractor : public interface::extractor<blob::decoded>
{
public:
    extractor(const blob::config& cfg) {}
    ~extractor() {}
    std::shared_ptr<blob::decoded> extract(const void* buf, size_t bufSize) const override
    {
        return std::make_shared<blob::decoded>(buf, bufSize);
    }

private:
};

class nervana::blob::loader : public interface::loader<blob::decoded>
{
public:
    loader(const blob::config& cfg) {}
    ~loader() {}
    void load(const std::vector<void*>& buflist, std::shared_ptr<blob::decoded> mp) const override
    {
        void* buf = buflist[0];
        memcpy(buf, mp->m_data, mp->m_data_size);
    }

private:
};
