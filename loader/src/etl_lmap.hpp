#pragma once

#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <opencv2/core/core.hpp>

#include "etl_interface.hpp"

namespace nervana {
    namespace lmap {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class box;
        class params;
        class config;
    }
}

class nervana::lmap::params : public nervana::params {
public:
    params() {}
};

class nervana::lmap::config : public interface::config {
public:
    config(nlohmann::json js);
    const std::vector<std::string> labels() const { return _labels; }

private:
    config() = delete;
    std::vector<std::string> _labels;
};

class nervana::lmap::decoded : public decoded_media {
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

class nervana::lmap::extractor : public nervana::interface::extractor<nervana::lmap::decoded> {
public:
    extractor( const nervana::lmap::config& );
    virtual ~extractor(){}
    virtual std::shared_ptr<nervana::lmap::decoded> extract(const char*, int) override;

    std::unordered_map<std::string,int>  get_data() { return _dictionary; }

private:
    std::unordered_map<std::string,int>  _dictionary;
};

class nervana::lmap::transformer : public nervana::interface::transformer<nervana::lmap::decoded, nervana::lmap::params> {
public:
    transformer();
    virtual ~transformer(){}
    virtual std::shared_ptr<nervana::lmap::decoded> transform(
                                            std::shared_ptr<nervana::lmap::params>,
                                            std::shared_ptr<nervana::lmap::decoded>) override;
private:
};

class nervana::lmap::loader : public nervana::interface::loader<nervana::lmap::decoded> {
public:
    loader();
    virtual ~loader(){}

    virtual void load(char*, std::shared_ptr<nervana::lmap::decoded>) override;
private:
};
