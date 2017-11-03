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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include "interface.hpp"
#include "image.hpp"
#include "util.hpp"
#include "augment_image.hpp"

namespace nervana
{
    namespace image
    {
        class config;
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }
    namespace video
    {
        class config; // Forward decl for friending
    }

    namespace multicrop
    {
        class config; // Forward decl for friending
    }
}

/**
 * \brief Configuration for image ETL
 *
 * An instantiation of this class controls the ETL of image data into the
 * target memory buffers from the source CPIO archives.
 */
class nervana::image::config : public interface::config
{
    friend class video::config;
    friend class multicrop::config;

public:
    uint32_t    height;
    uint32_t    width;
    std::string output_type{"uint8_t"};

    bool     channel_major = true;
    uint32_t channels      = 3;

    std::string name;

    config(nlohmann::json js);

    const std::vector<std::shared_ptr<interface::config_info_interface>>& get_config_list()
    {
        return config_list;
    }

    config() {}
private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(height, mode::REQUIRED),
        ADD_SCALAR(width, mode::REQUIRED),
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(channel_major, mode::OPTIONAL),
        ADD_SCALAR(channels, mode::OPTIONAL, [](uint32_t v) { return v == 1 || v == 3; }),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v) {
            return output_type::is_valid_type(v);
        })};

    void validate();

    bool flip_enable = false;
    bool center      = true;
};

// ===============================================================================================
// Decoded
// ===============================================================================================

class nervana::image::decoded : public interface::decoded_image
{
public:
    decoded() {}
    decoded(cv::Mat img) { _images.push_back(img); }
    bool add(cv::Mat img)
    {
        _images.push_back(img);
        return all_images_are_same_size();
    }
    bool add(const std::vector<cv::Mat>& images)
    {
        for (auto mat : images)
        {
            _images.push_back(mat);
        }
        return all_images_are_same_size();
    }
    virtual ~decoded() override {}
    cv::Mat& get_image(int index) { return _images[index]; }
    cv::Size2i             get_image_size() const { return _images[0].size(); }
    int                    get_image_channels() const { return _images[0].channels(); }
    size_t                 get_image_count() const { return _images.size(); }
    size_t                 get_size() const
    {
        return get_image_size().area() * get_image_channels() * get_image_count();
    }
    cv::Size2i image_size() const override { return _images[0].size(); }
protected:
    bool all_images_are_same_size()
    {
        for (int i = 1; i < _images.size(); i++)
        {
            if (_images[0].size() != _images[i].size())
                return false;
        }
        return true;
    }
    std::vector<cv::Mat> _images;
};

class nervana::image::extractor : public interface::extractor<image::decoded>
{
public:
    extractor(const image::config&);
    ~extractor() {}
    virtual std::shared_ptr<image::decoded> extract(const void*, size_t) const override;

    int get_channel_count() { return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1; }
private:
    int _pixel_type;
    int _color_mode;
};

class nervana::image::transformer
    : public interface::transformer<image::decoded, augment::image::params>
{
public:
    transformer(const image::config&);
    ~transformer() {}
    virtual std::shared_ptr<image::decoded>
        transform(std::shared_ptr<augment::image::params>,
                  std::shared_ptr<image::decoded>) const override;

    cv::Mat transform_single_image(std::shared_ptr<augment::image::params>, cv::Mat&) const;

private:
    image::photometric photo;
};

class nervana::image::loader : public interface::loader<image::decoded>
{
public:
    loader(const image::config& cfg, bool fixed_aspect_ratio);
    ~loader() {}
    virtual void load(const std::vector<void*>&, std::shared_ptr<image::decoded>) const override;

private:
    void split(cv::Mat&, char*);

    bool       m_channel_major;
    bool       m_fixed_aspect_ratio;
    shape_type m_stype;
    uint32_t   m_channels;
};
