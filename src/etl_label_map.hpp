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

#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <opencv2/core/core.hpp>

#include "interface.hpp"

namespace nervana
{
    namespace label_map
    {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class params;
        class config;
    }
}

class nervana::label_map::params : public interface::params
{
public:
    params() {}
};

class nervana::label_map::config : public interface::config
{
public:
    std::string              output_type = "uint32_t";
    std::vector<std::string> class_names;
    int                      max_classes = 100;
    std::string              name;

    config(nlohmann::json js);
    virtual ~config() {}
    int max_label_count() const { return max_classes; }
private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(output_type,
                   mode::OPTIONAL,
                   [](const std::string& v) { return output_type::is_valid_type(v); }),
        ADD_SCALAR(class_names, mode::REQUIRED),
        ADD_SCALAR(max_classes, mode::OPTIONAL)};

    config() {}
};

class nervana::label_map::decoded : public interface::decoded_media
{
    friend class label_map::extractor;

public:
    decoded();
    virtual ~decoded() {}
    const std::vector<int>& get_data() const { return labels; }
private:
    std::vector<int> labels;
};

class nervana::label_map::extractor : public interface::extractor<label_map::decoded>
{
public:
    extractor(const label_map::config&);
    virtual ~extractor() {}
    virtual std::shared_ptr<label_map::decoded> extract(const void*, size_t) const override;

    const std::unordered_map<std::string, int>& get_data() { return dictionary; }
private:
    std::unordered_map<std::string, int> dictionary;
};

class nervana::label_map::transformer
    : public interface::transformer<label_map::decoded, label_map::params>
{
public:
    transformer();
    virtual ~transformer() {}
    virtual std::shared_ptr<label_map::decoded>
        transform(std::shared_ptr<label_map::params>,
                  std::shared_ptr<label_map::decoded>) const override;

private:
};

class nervana::label_map::loader : public interface::loader<label_map::decoded>
{
public:
    loader(const label_map::config&);
    virtual ~loader() {}
    virtual void load(const std::vector<void*>&,
                      std::shared_ptr<label_map::decoded>) const override;

private:
    int max_label_count;
};
