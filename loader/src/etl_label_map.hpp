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
        config(nlohmann::json js);
        const std::vector<std::string> labels() const { return _label_list; }
        int max_label_count() const { return _max_label_count; }

    private:
        config() = delete;
        std::vector<std::string>    _label_list;
        int                         _max_label_count = 100;
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
