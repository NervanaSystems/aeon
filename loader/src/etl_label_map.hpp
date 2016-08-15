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
#include <opencv2/core/core.hpp>

#include "interface.hpp"

namespace nervana {
    namespace label_map {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class params;
        class config;
    }

    class label_map::params : public interface::params {
    public:
        params() {}
    };

    class label_map::config : public interface::config {
    public:
        std::string                 type_string = "uint32_t";
        std::vector<std::string>    labels;
        int                         max_labels = 100;

        config(nlohmann::json js);
        int max_label_count() const { return max_labels; }

    private:
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(type_string, mode::OPTIONAL),
            ADD_SCALAR(labels, mode::REQUIRED),
            ADD_SCALAR(max_labels, mode::OPTIONAL)
        };

        config() {}
    };

    class label_map::decoded : public interface::decoded_media {
        friend class transformer;
        friend class extractor;
    public:
        decoded();
        virtual ~decoded() {}

        const std::vector<int>& get_data() const { return _labels; }

    private:
        std::vector<int>    _labels;
    };

    class label_map::extractor : public interface::extractor<label_map::decoded> {
    public:
        extractor(const label_map::config&);
        virtual ~extractor(){}
        virtual std::shared_ptr<label_map::decoded> extract(const char*, int) override;

        std::unordered_map<std::string,int>  get_data() { return _dictionary; }

    private:
        std::unordered_map<std::string,int>  _dictionary;
    };

    class label_map::transformer : public interface::transformer<label_map::decoded, label_map::params> {
    public:
        transformer();
        virtual ~transformer(){}
        virtual std::shared_ptr<label_map::decoded> transform(
                                                std::shared_ptr<label_map::params>,
                                                std::shared_ptr<label_map::decoded>) override;
    private:
    };

    class label_map::loader : public interface::loader<label_map::decoded> {
    public:
        loader(const label_map::config&);
        virtual ~loader(){}

        virtual void load(const std::vector<void*>&, std::shared_ptr<label_map::decoded>) override;
    private:
        int max_label_count;
    };
}
