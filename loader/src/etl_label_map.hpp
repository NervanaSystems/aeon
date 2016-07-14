#pragma once

#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <opencv2/core/core.hpp>

#include "etl_interface.hpp"

namespace nervana {
    namespace label_map {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class box;
        class params;
        class config;
    }
}

class nervana::label_map::params : public nervana::params {
public:
    params() {}
};

class nervana::label_map::config : public interface::config {
public:
    config(nlohmann::json js);
    const std::vector<std::string> labels() const { return _label_list; }
    int max_label_count() const { return _max_label_count; }

private:
    config() = delete;
    std::vector<std::string>    _label_list;
    int                         _max_label_count = 100;
};

class nervana::label_map::decoded : public decoded_media {
    friend class transformer;
    friend class extractor;
public:
    decoded();
    virtual ~decoded() {}

    MediaType get_type() override { return MediaType::TARGET; }

    std::vector<int> get_data() const { return _labels; }

private:
    std::vector<int>    _labels;
};

class nervana::label_map::extractor : public nervana::interface::extractor<nervana::label_map::decoded> {
public:
    extractor(const nervana::label_map::config&);
    virtual ~extractor(){}
    virtual std::shared_ptr<nervana::label_map::decoded> extract(const char*, int) override;

    std::unordered_map<std::string,int>  get_data() { return _dictionary; }

private:
    std::unordered_map<std::string,int>  _dictionary;
};

class nervana::label_map::transformer : public nervana::interface::transformer<nervana::label_map::decoded, nervana::label_map::params> {
public:
    transformer();
    virtual ~transformer(){}
    virtual std::shared_ptr<nervana::label_map::decoded> transform(
                                            std::shared_ptr<nervana::label_map::params>,
                                            std::shared_ptr<nervana::label_map::decoded>) override;
private:
};

class nervana::label_map::loader : public nervana::interface::loader<nervana::label_map::decoded> {
public:
    loader(const nervana::label_map::config&);
    virtual ~loader(){}

    virtual void load(char*, std::shared_ptr<nervana::label_map::decoded>) override;
private:
    int max_label_count;
};
