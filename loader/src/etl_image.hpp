#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_interface.hpp"
#include "params.hpp"

namespace nervana {


    class image_settings : public parameter_collection {
    public:
        cv::Rect cropbox;
        cv::Size size;
        int angle;
        bool flip;
        float colornoise[3];  //pixelwise random values
        float cbs[3];  // contrast, brightness, saturation
    };


    class decoded_image : public decoded_media {
    public:
        decoded_image() {}
        decoded_image(cv::Mat img) { _img = img; }
        virtual ~decoded_image() override {}

        inline MediaType get_type() override { return MediaType::IMAGE; }
        inline cv::Mat& get_image() { return _img; }

    private:
        cv::Mat _img;
    };


    class image_extractor : public extractor_interface {
    public:
        image_extractor(param_ptr);
        ~image_extractor() {}
        virtual media_ptr extract(char*, int) override;

        const int get_channel_count() {return _channel_count;}
    private:
        int _channel_count;
        int _pixel_type;
        int _color_mode;
    };


    class image_transformer : public transformer_interface {
    public:
        image_transformer(param_ptr);
        ~image_transformer() {}
        virtual media_ptr transform(settings_ptr, const media_ptr&) override;
        virtual void fill_settings(settings_ptr) override;

    private:
        void rotate(const cv::Mat& input, cv::Mat& output, int angle);
        void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
        void lighting(cv::Mat& inout, float pixelstd[]);
        void cbsjitter(cv::Mat& inout, float cbs[]);

    };


    class image_loader : public loader_interface {
    public:
        image_loader(param_ptr);
        ~image_loader() {}
        virtual void load(char*, int, const media_ptr&) override;

    private:
        void split(cv::Mat& img, char* buf, int bufSize);
    };

}
