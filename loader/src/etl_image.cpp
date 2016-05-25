#include "etl_image.hpp"

namespace nervana {

    /* Extract */
    image_extractor::image_extractor(param_ptr_t image_extractor_params)
    {
        _channel_count = image_extractor_params->get_channel_count();
        if !(_channel_count == 1 || _channel_count == 3)
        {
            std::stringstream ss;
            ss << "Unsupported number of channels in image: " << _channel_count;
            throw std::runtime_error(ss.str());
        } else {
            _pixel_type = _channel_count == 1 ? CV_8UC1 : CV_8UC3;
            _color_mode = _channel_count == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
        }

    }

    media_ptr_t image_extractor::extract(char* inbuf, int insize)
    {
        auto output = make_shared<decoded_image>();
        cv::Mat input_img(1, insize, _pixel_type, inbuf);
        cv::imdecode(input_img, _color_mode, output->get_image());
        return static_pointer_cast<decoded_media>(output);
    }




    image_transformer::image_transformer(param_ptr_t image_transformer_params)
    {


    }

    media_ptr_t image_transformer::transform(settings_ptr_t transform_settings,
                                             const media_ptr_t& input)
    {
        fill_settings(transform_settings)

        cv::Mat rotatedImage;
        auto img = static_pointer_cast<decoded_image>(input);
        rotate(img->get_image(), rotatedImage, transform_settings->get_angle());
        cv::Mat croppedImage = rotatedImage(transform_settings->get_cropbox());

        cv::Mat resizedImage;
        resize(croppedImage, resizedImage, transform_settings->get_size());
        cbsjitter(croppedImage, transform_settings->get_cbs());
        lighting(croppedImage, transform_settings->get_colornoise());

        cv::Mat *finalImage = &resizedImage;
        cv::Mat flippedImage;
        if (transform_settings->get_flip()) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        }

        auto output = make_shared<decoded_image>(*finalImage);
        return static_pointer_cast<decoded_media>(output);
    }

    void image_transformer::fill_settings(settings_ptr_t settings)
    {
        if (settings->filled())
        {
            return;
        }

    }

    void image_transformer::rotate(const cv::Mat& input, cv::Mat& output, int angle)
    {
        if (angle == 0) {
            output = input;
        } else {
            cv::Point2i pt(input.cols / 2, input.rows / 2);
            cv::Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
            cv::warpAffine(input, output, rot, input.size());
        }
    }

    void image_transformer::resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size)
    {
        if (size == input.size()) {
            output = input;
        } else {
            int inter = input.size().area() < size.area() ? CV_INTER_CUBIC : CV_INTER_AREA;
            cv::resize(input, output, size, 0, 0, inter);
        }
    }

    void image_transformer::lighting(cv::Mat& inout, float pixelstd[])
    {
    }

    void image_transformer::cbsjitter(cv::Mat& inout, float cbs[])
    {
    }





    image_loader::image_loader(param_ptr_t image_loader_params) {}

    void image_loader::load(char* outbuf, int outsize, const media_ptr_t& input)
    {
        auto img = static_pointer_cast<decoded_image>(input);
        this->split(img->get_image(), outbuf, outsize);
    }

    void image_loader::split(cv::Mat& img, char* buf, int bufSize)
    {
        int pix_per_channel = img.total();
        int num_channels = img.channels();
        int all_pixels = pix_per_channel * num_channels;

        if (all_pixels > bufSize) {
            throw std::runtime_error("Decode failed - buffer too small");
        }

        if (num_channels == 1) {
            memcpy(buf, img.data, all_pixels);
            return;
        }

        // Split into separate channels
        cv::Size2i size = img.size();
        cv::Mat b(size, CV_8U, buf);
        cv::Mat g(size, CV_8U, buf + pix_per_channel);
        cv::Mat r(size, CV_8U, buf + 2 * pix_per_channel);

        cv::Mat channels[3] = {b, g, r};
        cv::split(img, channels);
    }


}
