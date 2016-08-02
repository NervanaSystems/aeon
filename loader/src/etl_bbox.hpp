#pragma once

#include <string>
#include <unordered_map>

#include "interface.hpp"
#include "etl_image.hpp"
#include "json.hpp"
#include "box.hpp"

namespace nervana {
    namespace bbox {
        class decoded;
        class config;
        class extractor;
        class transformer;
        class loader;
        class box;
    }
}

class nervana::bbox::box : public nervana::box {
public:
    bool difficult = false;
    bool truncated = false;
    int label;
};

std::ostream& operator<<(std::ostream&,const nervana::bbox::box&);

class nervana::bbox::config : public nervana::interface::config {
public:
    size_t                      height;
    size_t                      width;
    size_t                      max_bbox_count;
    std::vector<std::string>    labels;
    std::string                 type_string = "float";

    std::unordered_map<std::string,int> label_map;

    config(nlohmann::json js, bool ignore_errors=false);

private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(height, mode::REQUIRED),
        ADD_SCALAR(width, mode::REQUIRED),
        ADD_SCALAR(max_bbox_count, mode::REQUIRED),
        ADD_SCALAR(labels, mode::REQUIRED),
        ADD_SCALAR(type_string, mode::OPTIONAL)
    };

    config() = delete;
    void validate();
};

class nervana::bbox::decoded : public interface::decoded_media {
    friend class transformer;
    friend class extractor;
public:
    decoded();
    bool extract(const char* data, int size, const std::unordered_map<std::string,int>& label_map);
    virtual ~decoded() {}

    const std::vector<box>& boxes() const { return _boxes; }
    int width() const { return _width; }
    int height() const { return _height; }
    int depth() const { return _depth; }

private:
    std::vector<box> _boxes;
    int _width;
    int _height;
    int _depth;
};


class nervana::bbox::extractor : public nervana::interface::extractor<nervana::bbox::decoded> {
public:
    extractor(const std::unordered_map<std::string,int>&);
    virtual ~extractor(){}
    virtual std::shared_ptr<bbox::decoded> extract(const char*, int) override;
    void extract(const char*, int, std::shared_ptr<bbox::decoded>&);

private:
    extractor() = delete;
    std::unordered_map<std::string,int> label_map;
};

class nervana::bbox::transformer : public nervana::interface::transformer<nervana::bbox::decoded, nervana::image::params> {
public:
    transformer(const bbox::config&);
    virtual ~transformer(){}
    virtual std::shared_ptr<bbox::decoded> transform(
                                            std::shared_ptr<image::params>,
                                            std::shared_ptr<bbox::decoded>) override;

private:
};

class nervana::bbox::loader : public nervana::interface::loader<nervana::bbox::decoded> {
public:
    loader(const bbox::config&);
    virtual ~loader(){}
    virtual void load(char*, std::shared_ptr<bbox::decoded>) override;

private:
    const size_t max_bbox;
};
