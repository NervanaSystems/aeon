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

#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <cstdint>

#include "interface.hpp"

namespace nervana
{
    namespace char_map
    {
        class decoded;
        class extractor;
        class loader;
        class config;

        using cmap_t = std::unordered_map<wchar_t, uint32_t>;
    }
}

class nervana::char_map::config : public interface::config
{
    friend class extractor;

public:
    /** Maximum length of each transcript. Samples with longer transcripts
    * will be truncated */
    uint32_t max_length;
    /** Character map alphabet */
    std::string alphabet;
    /** Same alphabet in wchar_t */
    std::wstring walphabet;
    /** Integer value to give to unknown characters. 0 causes them to be
    * discarded.*/
    uint32_t unknown_value = 0;
    /** Whether to also output the number of valid characters per record */
    bool emit_length = false;
    /** Output data type */
    std::string output_type{"uint32_t"};
    std::string name;

    config(nlohmann::json js);

    const cmap_t& get_cmap() const { return _cmap; }
private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(max_length, mode::REQUIRED),
        ADD_SCALAR(alphabet, mode::REQUIRED),
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(unknown_value, mode::OPTIONAL),
        ADD_SCALAR(emit_length, mode::OPTIONAL),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v) {
            return output_type::is_valid_type(v);
        })};
    cmap_t _cmap;

    config() {}
    void validate()
    {
        if (output_type != "uint32_t")
        {
            throw std::runtime_error("Invalid load type for char map " + output_type);
        }
        if (!unique_chars(walphabet))
        {
            throw std::runtime_error("alphabet does not consist of unique chars " + alphabet);
        }
        if (unknown_value > 0 && unknown_value < walphabet.size())
        {
            throw std::runtime_error(
                "unknown_value should be >= alphabet length and <= 4294967295");
        }
    }

    bool unique_chars(std::wstring test_string)
    {
        if (test_string.size() > UINT32_MAX)
        {
            return false;
        }

        std::sort(test_string.begin(), test_string.end());

        for (uint32_t i = 1; i < test_string.size(); i++)
        {
            if (test_string[i - 1] == test_string[i])
            {
                return false;
            }
        }
        return true;
    }
};

class nervana::char_map::decoded : public interface::decoded_media
{
public:
    decoded(std::vector<uint32_t> char_ints, uint32_t nvalid)
        : _labels{char_ints}
        , _nvalid{nvalid}
    {
    }

    virtual ~decoded() {}
    std::vector<uint32_t> get_data() const { return _labels; }
    uint32_t              get_length() const { return _nvalid; }
private:
    std::vector<uint32_t> _labels;
    uint32_t              _nvalid;
};

class nervana::char_map::extractor : public interface::extractor<char_map::decoded>
{
public:
    extractor(const char_map::config& cfg)
        : _cmap{cfg.get_cmap()}
        , _max_length{cfg.max_length}
        , _unknown_value{cfg.unknown_value}
    {
    }

    virtual ~extractor() {}
    // in_array is transcription in UTF-8
    // in_sz is max size of transcription in unicode characters
    virtual std::shared_ptr<char_map::decoded> extract(const void* in_array,
                                                       size_t      in_sz) const override;

private:
    const cmap_t&  _cmap; // This comes from config
    uint32_t       _max_length;
    const uint32_t _unknown_value;
};

class nervana::char_map::loader : public interface::loader<char_map::decoded>
{
public:
    loader(const char_map::config& cfg)
        : _emit_length(cfg.emit_length)
    {
    }
    virtual ~loader() {}
    virtual void load(const std::vector<void*>&, std::shared_ptr<char_map::decoded>) const override;

private:
    const bool _emit_length;
};
