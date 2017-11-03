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
    namespace label
    {
        class config;
        class decoded;

        class extractor;
        class loader;
    }
}

class nervana::label::config : public interface::config
{
public:
    bool        binary = false;
    std::string output_type{"uint32_t"};
    std::string type;
    std::string name;

    config(nlohmann::json js)
    {
        for (auto& info : config_list)
        {
            info->parse(js);
        }
        verify_config("label", config_list, js);

        add_shape_type({1}, output_type);
    }

private:
    config() {}
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(binary, mode::OPTIONAL),
        ADD_SCALAR(type, mode::OPTIONAL),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v) {
            return output_type::is_valid_type(v);
        })};
};

class nervana::label::decoded : public interface::decoded_media
{
public:
    decoded(int index)
        : _index{index}
    {
    }
    virtual ~decoded() override {}
    int get_index() { return _index; }
private:
    decoded() = delete;
    int _index;
};

class nervana::label::extractor : public interface::extractor<label::decoded>
{
public:
    extractor(const label::config& cfg)
        : _binary{cfg.binary}
    {
    }

    ~extractor() {}
    std::shared_ptr<label::decoded> extract(const void* buf, size_t bufSize) const override
    {
        int lbl;
        if (_binary)
        {
            if (bufSize != 4)
            {
                std::stringstream ss;
                ss << "Only 4 byte buffers can be loaded as int32.  ";
                ss << "label_extractor::extract received " << bufSize << " bytes";
                throw std::runtime_error(ss.str());
            }
            lbl = unpack<int>((const char*)buf);
        }
        else
        {
            try
            {
                lbl = std::stoi(std::string((const char*)buf, bufSize));
            }
            catch (const std::invalid_argument& ex)
            {
                ERR << "Cannot convert string to integer: " << ex.what();
                throw ex;
            }
            catch (const std::out_of_range& ex)
            {
                ERR << "String to int conversion out of range error: " << ex.what();
                throw ex;
            }
        }
        return std::make_shared<label::decoded>(lbl);
    }

private:
    bool _binary;
};

class nervana::label::loader : public interface::loader<label::decoded>
{
public:
    loader(const label::config& cfg)
        : _cfg{cfg}
    {
    }
    ~loader() {}
    void load(const std::vector<void*>& buflist, std::shared_ptr<label::decoded> mp) const override
    {
        char* buf   = reinterpret_cast<char*>(buflist[0]);
        int   index = mp->get_index();
        memcpy(buf, &index, _cfg.get_shape_type().get_otype().get_size());
    }

private:
    const label::config& _cfg;
};
