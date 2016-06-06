#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_interface.hpp"
#include "params.hpp"


namespace nervana {
    namespace image {
        class extract_params;
        class transform_params;
        class load_params;

        class extractor;
        class transformer;
        class loader;

        class settings;
    }
    class decoded_images;
}

class nervana::image::extract_params : public nervana::json_parameter_collection {
public:
    int num_channels = 3;
    extract_params(std::string argString) {
        auto js = nlohmann::json::parse(argString);

        parse_opt(num_channels, "num_channels", js);
    }
};


class nervana::image::transform_params : public nervana::json_parameter_collection {
public:
    int height;
    int width;
    std::uniform_real_distribution<float> scale{1.0f, 1.0f};

    std::uniform_int_distribution<int>    angle{0, 0};

    std::normal_distribution<float>       lighting{0.0f, 0.0f};
    std::uniform_real_distribution<float> aspect_ratio{1.0f, 1.0f};
    std::uniform_real_distribution<float> photometric{0.0f, 0.0f};
    std::uniform_real_distribution<float> crop_offset{1.0f, 1.0f};
    std::bernoulli_distribution           flip{0};

    bool do_area_scale = false;

    transform_params(std::string argString) {
        auto js = nlohmann::json::parse(argString);

        parse_req(height, "height", js);
        parse_req(width, "width", js);

        parse_opt(do_area_scale, "do_area_scale", js);

        parse_dist<decltype(angle)>(angle, "angle_dist_params", js);
        parse_dist<decltype(scale)>(scale, "scale_dist_params", js);
        parse_dist<decltype(lighting)>(lighting, "lighting_dist_params", js);
        parse_dist<decltype(aspect_ratio)>(aspect_ratio, "aspect_ratio_dist_params", js);
        parse_dist<decltype(photometric)>(photometric, "photometric_dist_params", js);
        parse_dist<decltype(crop_offset)>(crop_offset, "crop_offset_dist_params", js);
        parse_dist<decltype(flip)>(flip, "flip_dist_params", js);
    }

};

class nervana::image::load_params : public nervana::json_parameter_collection {
public:
    bool channel_major = true;

    load_params(std::string argString) {
        auto js = nlohmann::json::parse(argString);
        parse_opt(channel_major, "channel_major", js);
    }
};


class nervana::image::settings : public nervana::settings {
public:

    settings() {}
    void dump();

    cv::Rect cropbox;
    int angle = 0;
    bool flip = false;
    std::vector<float> lighting{0.0, 0.0, 0.0};  //pixelwise random values
    std::vector<float> photometric{0.0, 0.0, 0.0};  // contrast, brightness, saturation
    bool filled = false;
};


class nervana::decoded_images : public nervana::decoded_media {
public:
    decoded_images() {}
    decoded_images(cv::Mat img) : _img(img) { _images.push_back(_img); }
    virtual ~decoded_images() override {}

    virtual MediaType get_type() override { return MediaType::IMAGE; }
    cv::Mat& get_image(int index) { return _images[index]; }
    cv::Size2i get_image_size() {return _images[0].size(); }
    size_t size() const { return _images.size(); }

private:
    cv::Mat _img;
    std::vector<cv::Mat> _images;
};


class nervana::image::extractor : public nervana::interface::extractor {
public:
    extractor(param_ptr);
    ~extractor() {}
    virtual media_ptr extract(char*, int) override;

    const int get_channel_count() {return _channel_count;}
private:
    int _channel_count;
    int _pixel_type;
    int _color_mode;
};


class nervana::image::transformer : public nervana::interface::transformer {
public:
    transformer(param_ptr);
    ~transformer() {}
    virtual media_ptr transform(settings_ptr, const media_ptr&) override;
    virtual void fill_settings(settings_ptr, const media_ptr&, std::default_random_engine &) override;

private:
    void rotate(const cv::Mat& input, cv::Mat& output, int angle);
    void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
    void lighting(cv::Mat& inout, std::vector<float>);
    void cbsjitter(cv::Mat& inout, std::vector<float>);

    std::shared_ptr<transform_params> _itp;
};


class nervana::image::loader : public nervana::interface::loader {
public:
    loader(param_ptr);
    ~loader() {}
    virtual void load(char*, int, const media_ptr&) override;

private:
    void split(cv::Mat& img, char* buf, int bufSize);
};

