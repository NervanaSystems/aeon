#include "etl_image.hpp"

using namespace std;

namespace nervana {

    /* Extract */
    image_extractor::image_extractor(param_ptr pptr)
    {
        auto image_extractor_params = static_pointer_cast<image_params>(pptr);

        _channel_count = image_extractor_params->channels;
        if (!(_channel_count == 1 || _channel_count == 3))
        {
            std::stringstream ss;
            ss << "Unsupported number of channels in image: " << _channel_count;
            throw std::runtime_error(ss.str());
        } else {
            _pixel_type = _channel_count == 1 ? CV_8UC1 : CV_8UC3;
            _color_mode = _channel_count == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
        }

    }

    media_ptr image_extractor::extract(char* inbuf, int insize)
    {
        cv::Mat output_img;
        cv::Mat input_img(1, insize, _pixel_type, inbuf);
        cv::imdecode(input_img, _color_mode, &output_img);

        return make_shared<decoded_image>(output_img);
    }


    /* Transform:
        transformer_params will be a supplied bunch of settings used by this provider.
        on each record, the transformer will use the transform_params along with the supplied
        record to fill a transform_settings structure which will have

        Spatial distortion settings:
        randomly sampled crop box (based on params->center, params->aspect_ratio, params->scale_pct, record size)
        randomly determined flip (based on params->flip)
        randomly sampled rotation angle (based on params->angle)

        Photometric distortion settings:
        randomly sampled contrast, brightness, saturation, lighting values (based on params->cbs, lighting bounds)

    */
    image_transformer::image_transformer(param_ptr image_transformer_params)
    {
        // should we be doing anything in here?
    }

    media_ptr image_transformer::transform(settings_ptr transform_settings,
                                           const media_ptr& input)
    {
        auto img_xform = static_pointer_cast<image_settings>(transform_settings);
        fill_settings(img_xform);

        cv::Mat rotatedImage;
        auto img = static_pointer_cast<decoded_image>(input);
        rotate(img->get_image(), rotatedImage, img_xform->angle);
        cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

        cv::Mat resizedImage;
        resize(croppedImage, resizedImage, img_xform->size);
        cbsjitter(croppedImage, img_xform->cbs);
        lighting(croppedImage, img_xform->colornoise);

        cv::Mat *finalImage = &resizedImage;
        cv::Mat flippedImage;
        if (img_xform->flip) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        }

        return make_shared<decoded_image>(*finalImage);
    }

    void image_transformer::fill_settings(settings_ptr settings)
    {
        // if (settings->filled())
        // {
        //     return;
        // }
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





    image_loader::image_loader(param_ptr image_loader_params) {}

    void image_loader::load(char* outbuf, int outsize, const media_ptr& input)
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
