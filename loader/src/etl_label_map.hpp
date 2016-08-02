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

        config() = delete;
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

        virtual void load(char*, std::shared_ptr<label_map::decoded>) override;
    private:
        int max_label_count;
    };
}
