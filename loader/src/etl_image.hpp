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

class nervana::image::extract_params : public nervana::parameter_collection {
public:
    int channels;
    extract_params() {
        ADD_OPTIONAL(channels, "number of channels", "ch", "channels", 3, 1, 3);
    }
};


class nervana::image::transform_params : public nervana::parameter_collection {
public:
    int height;
    int width;
    float scale_pct;
    int angle;
    float cbs_range;
    float lighting_range;
    float aspect_ratio;
    bool area_scaling;
    bool flip;
    float crop_offset_range;

    transform_params() {
        // Required Params
        ADD_REQUIRED(height, "image height", "h", "height");
        ADD_REQUIRED(width, "image width", "w", "width");
        ADD_REQUIRED(scale_pct, "percentage of original image to scale crop", "s1", "scale_pct", 1, 100);

        ADD_OPTIONAL(angle, "rotation angle", "angle", "rotate_angle", 0, 0, 90);
        ADD_OPTIONAL(cbs_range, "augmentation range in pct to jitter contrast, brightness, saturation", "c1", "cbs_range", 0.0, 0.0, 1.0);
        ADD_OPTIONAL(lighting_range, "augmentation range in pct to jitter lighting", "l1", "lighting_range", 0.0, 0.0, 0.2);
        ADD_OPTIONAL(aspect_ratio, "aspect ratio to jitter", "a1", "aspect_ratio", 1.0, 1.0, 2.0);
        ADD_OPTIONAL(area_scaling, "whether to use area based scaling", "a2", "area_scaling", false);
        ADD_OPTIONAL(flip, "randomly flip?", "f1", "flip", false);
        ADD_OPTIONAL(crop_offset_range, "augmentation range in pct to cropbox location", "c2", "crop_offset_range", 0.0, 0.0, 1.0);

        _rngn = std::normal_distribution<float>(0.0, 1.0);
        _rngu = std::uniform_real_distribution<float>(-1.0, 1.0);
    }

    void fill_settings(media_ptr, settings_ptr, std::default_random_engine);

private:
    std::uniform_real_distribution<float>    _rngu;
    std::normal_distribution<float>          _rngn;

};


class nervana::image::load_params : public nervana::parameter_collection {
public:
    bool channel_major;

    load_params() {
        ADD_OPTIONAL(channel_major, "load in channel major mode?", "cm", "channel_major", true);
    }
};


class nervana::image::settings : public nervana::settings {
public:

    settings() {}
    cv::Rect cropbox;
    int angle = 0;
    bool flip = false;
    float colornoise[3] {0.0, 0.0, 0.0};  //pixelwise random values
    float cbs[3] {0.0, 0.0, 0.0};  // contrast, brightness, saturation
    bool filled = false;
};


class nervana::decoded_images : public nervana::decoded_media {
public:
    decoded_images() {}
    decoded_images(cv::Mat img) : _img(img) { _images.push_back(_img); }
    virtual ~decoded_images() override {}

    virtual MediaType get_type() override { return MediaType::IMAGE; }
    cv::Mat& get_image(int index) { return _images[index]; }
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

private:
    void rotate(const cv::Mat& input, cv::Mat& output, int angle);
    void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
    void lighting(cv::Mat& inout, float pixelstd[]);
    void cbsjitter(cv::Mat& inout, float cbs[]);

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

