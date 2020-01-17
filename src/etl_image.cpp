/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "etl_image.hpp"
#ifdef PYTHON_PLUGIN
#include "python_plugin.hpp"
#endif

#include <atomic>

using namespace std;
using namespace nervana;

image::config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw runtime_error("missing image config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("image", config_list, js);

    if (channel_major)
    {
        add_shape_type({channels, height, width}, {"channels", "height", "width"}, output_type);
    }
    else
    {
        add_shape_type({height, width, channels}, {"height", "width", "channels"}, output_type);
    }

    validate();
}

void image::config::validate()
{
    if (width <= 0)
    {
        throw invalid_argument("invalid width");
    }
    if (height <= 0)
    {
        throw invalid_argument("invalid height");
    }
    if (bgr_to_rgb && channels != 3)
    {
        throw invalid_argument(
            "invalid config: bgr_to_rgb can be 'true' only for channels set to '3'");
    }
}

/* Extract */
image::extractor::extractor(const image::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw runtime_error(ss.str());
    }
    else
    {
        _pixel_type = CV_MAKETYPE(CV_8U, cfg.channels);
        _color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

shared_ptr<image::decoded> image::extractor::extract(const void* inbuf, size_t insize) const
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, _pixel_type, (char*)inbuf);
    cv::imdecode(input_img, _color_mode, &output_img);

    if (output_img.empty()) {
        throw runtime_error("Decoding image failed due to invalid data in the image file.");
    }

    auto rc = make_shared<image::decoded>();
    rc->add(output_img); // don't need to check return for single image
    return rc;
}

/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly sampled crop box (based on params->center, params->horizontal_distortion, params->scale_pct, record size)
    randomly determined flip (based on params->flip)
    randomly sampled rotation angle (based on params->angle)

    Photometric distortion params:
    randomly sampled contrast, brightness, saturation, lighting values (based on params->cbs, lighting bounds)

*/
image::transformer::transformer(const image::config&)
{
}

shared_ptr<image::decoded>
    image::transformer::transform(shared_ptr<augment::image::params> img_xform,
                                  shared_ptr<image::decoded>         img) const
{
    vector<cv::Mat> finalImageList;
    for (int i = 0; i < img->get_image_count(); i++)
    {
        finalImageList.push_back(transform_single_image(img_xform, img->get_image(i)));
    }

    auto rc = make_shared<image::decoded>();
    if (rc->add(finalImageList) == false)
    {
        rc = nullptr;
    }
    return rc;
}

/**
 * rotate
 * expand
 * resize_short
 * crop
 * resize
 * distort
 * flip
 */
cv::Mat image::transformer::transform_single_image(shared_ptr<augment::image::params> img_xform,
                                                   cv::Mat& single_img) const
{
    // img_xform->dump(cout);
    cv::Mat rotatedImage;
    image::rotate(single_img, rotatedImage, img_xform->angle);

    cv::Mat expandedImage;
    if (img_xform->expand_ratio > 1.0)
        image::expand(
            rotatedImage, expandedImage, img_xform->expand_offset, img_xform->expand_size);
    else
        expandedImage = rotatedImage;

    // TODO(sfraczek): add test for this resize short
    cv::Mat resizedShortImage;
    if (img_xform->resize_short_size == 0)
        resizedShortImage = expandedImage;
    else
        image::resize_short(expandedImage,
                            resizedShortImage,
                            img_xform->resize_short_size,
                            img_xform->interpolation_method);

    cv::Mat croppedImage = resizedShortImage(img_xform->cropbox);
    image::add_padding(croppedImage, img_xform->padding, img_xform->padding_crop_offset);

    cv::Mat resizedImage;
    image::resize(croppedImage,
                  resizedImage,
                  img_xform->output_size,
                  img_xform->interpolation_method);
    photo.cbsjitter(resizedImage,
                    img_xform->contrast,
                    img_xform->brightness,
                    img_xform->saturation,
                    img_xform->hue);
    photo.lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

    cv::Mat flippedImage;

    if (img_xform->flip)
    {
        cv::flip(resizedImage, flippedImage, 1);
    }
    else
        flippedImage = resizedImage;

    cv::Mat* finalImage = &flippedImage;

#ifdef PYTHON_PLUGIN
    cv::Mat pluginImage;
    if (img_xform->user_plugin)
    {
        pluginImage = img_xform->user_plugin->augment_image(flippedImage);
        finalImage  = &pluginImage;
    }
#endif

    return *finalImage;
}

image::loader::loader(const image::config& cfg,
                      bool                 fixed_aspect_ratio,
                      std::vector<double>  mean,
                      std::vector<double>  stddev)
    : m_channel_major{cfg.channel_major}
    , m_fixed_aspect_ratio{fixed_aspect_ratio}
    , m_bgr_to_rgb{cfg.bgr_to_rgb}
    , m_stype{cfg.get_shape_type()}
    , m_channels{cfg.channels}
    , m_mean{mean}
    , m_stddev{stddev}
{
    if (!m_mean.empty() || !m_stddev.empty())
    {
        if (!(cfg.output_type.compare("float") == 0 || cfg.output_type.compare("double") == 0))
        {
            throw invalid_argument(
                "Standardization (mean, stddev) is supported only for float or double "
                "'output_type'.");
        }
        if (m_mean.size() != cfg.channels || m_stddev.size() != cfg.channels)
        {
            throw invalid_argument(
                "Size of 'mean' and 'stddev' must be equal to number of channels or empty.");
        }
    }

    if (m_bgr_to_rgb)
    {
        m_from_to = {0, 2, 1, 1, 2, 0};
    }
    else
    {
        m_from_to.reserve(m_channels * 2);
        for (int i = 0; i < m_channels; i++)
        {
            m_from_to.push_back(i);
            m_from_to.push_back(i);
        }
    }
}

void image::loader::load(const vector<void*>& outlist, shared_ptr<image::decoded> input) const
{
    char* outbuf = (char*)outlist[0];
    // TODO: Generalize this to also handle multi_crop case
    auto cv_type      = m_stype.get_otype().get_cv_type();
    auto element_size = m_stype.get_otype().get_size();
    // if m_channels is 3 but images has 1 channel it is converted to
    // 3 channels so we need m_channels instead of input channels
    int image_size = m_channels * input->get_image(0).total() * element_size;

    for (int i = 0; i < input->get_image_count(); i++)
    {
        auto outbuf_i    = outbuf + (i * image_size);
        auto input_image = input->get_image(i);

        vector<cv::Mat> source;
        vector<cv::Mat> target;

        if (m_fixed_aspect_ratio)
        {
            // zero out the output buffer as the image may not fill the canvas
            std::fill_n(outbuf_i, m_stype.get_byte_size(), 0);

            vector<size_t> shape = m_stype.get_shape();
            // methods for image_var
            if (m_channel_major)
            {
                // Split into separate channels
                int      height          = shape[1];
                int      width           = shape[2];
                int      pix_per_channel = width * height;
                cv::Mat  b(height, width, CV_8U, outbuf_i);
                cv::Mat  g(height, width, CV_8U, outbuf_i + pix_per_channel);
                cv::Mat  r(height, width, CV_8U, outbuf_i + 2 * pix_per_channel);
                cv::Rect roi(0, 0, input_image.cols, input_image.rows);
                cv::Mat  b_roi = b(roi);
                cv::Mat  g_roi = g(roi);
                cv::Mat  r_roi = r(roi);
                // TODO(sfraczek): unify this to mix_channels.
                //  split will fail for 1 channel input image
                std::vector<cv::Mat> channels;
                if (m_bgr_to_rgb)
                {
                    channels = {r_roi, g_roi, b_roi};
                    cv::split(input_image, channels);
                }
                else
                {
                    channels = {b_roi, g_roi, r_roi};
                    cv::split(input_image, channels);
                }
                // channelwise call
                if (!m_mean.empty())
                    image::standardize(channels, m_mean, m_stddev);
            }
            else
            {
                int     channels = shape[2];
                int     width    = shape[1];
                int     height   = shape[0];
                cv::Mat output(height, width, CV_8UC(channels), outbuf_i);
                cv::Mat target_roi = output(cv::Rect(0, 0, input_image.cols, input_image.rows));
                source.push_back(input_image);
                target.push_back(target_roi);
                image::convert_mix_channels(source, target, m_from_to, m_bgr_to_rgb);
                // single image call
                if (!m_mean.empty())
                    image::standardize(target, m_mean, m_stddev);
            }
        }
        else
        {
            // methods for image
            source.push_back(input_image);
            if (m_channel_major)
            {
                for (int ch = 0; ch < m_channels; ch++)
                {
                    target.emplace_back(
                        input_image.size(),
                        cv_type,
                        (char*)(outbuf_i + ch * input_image.total() * element_size));
                }
            }
            else
            {
                target.emplace_back(
                    input_image.size(), CV_MAKETYPE(cv_type, m_channels), (char*)(outbuf_i));
            }
            image::convert_mix_channels(source, target, m_from_to, m_bgr_to_rgb);
            // single image call
            if (!m_mean.empty())
                image::standardize(target, m_mean, m_stddev);
        }
    }
}
