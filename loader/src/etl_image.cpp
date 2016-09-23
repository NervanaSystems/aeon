/*
 Copyright 2016 Nervana Systems Inc.
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

#include "etl_image.hpp"

using namespace std;
using namespace nervana;

image::config::config(nlohmann::json js)
{
    if(js.is_null()) {
        throw std::runtime_error("missing image config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config("image", config_list, js);

    // Now fill in derived
    shape_t shape;
    if (flip_enable) {
        flip_distribution = bernoulli_distribution{0.5};
    }

    if (!center) {
        crop_offset = uniform_real_distribution<float> {0.0f, 1.0f};
    }

    if (channel_major) {
        shape = {channels, height, width};
    } else{
        shape = {height, width, channels};
    }
    add_shape_type(shape, output_type);

    validate();
}

void image::config::validate() {
    if(crop_offset.param().a() > crop_offset.param().b()) {
        throw std::invalid_argument("invalid crop_offset");
    }
    if(width <= 0) {
        throw std::invalid_argument("invalid width");
    }
    if(height <= 0) {
        throw std::invalid_argument("invalid height");
    }
}

void image::params::dump(ostream& ostr)
{
    ostr << "cropbox             " << cropbox                 << "\n";
    ostr << "output_size         " << output_size             << "\n";
    ostr << "angle               " << angle                   << "\n";
    ostr << "flip                " << flip                    << "\n";
    ostr << "lighting            " << join(lighting, ", ")    << "\n";
    ostr << "color_noise_std     " << color_noise_std         << "\n";
    ostr << "contrast            " << contrast                << "\n";
    ostr << "brightness          " << brightness              << "\n";
    ostr << "saturation          " << saturation              << "\n";
    ostr << "hue                 " << hue                     << "\n";
    ostr << "debug_deterministic " << debug_deterministic     << "\n";
}


/* Extract */
image::extractor::extractor(const image::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw std::runtime_error(ss.str());
    } else {
        _pixel_type = CV_MAKETYPE(CV_8U, cfg.channels);
        _color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

shared_ptr<image::decoded> image::extractor::extract(const char* inbuf, int insize)
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, _pixel_type, const_cast<char*>(inbuf));
    cv::imdecode(input_img, _color_mode, &output_img);

    auto rc = make_shared<image::decoded>();
    rc->add(output_img);    // don't need to check return for single image
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

shared_ptr<image::decoded> image::transformer::transform(
                                                 shared_ptr<image::params> img_xform,
                                                 shared_ptr<image::decoded> img)
{
    vector<cv::Mat> finalImageList;
    for(int i=0; i<img->get_image_count(); i++) {
        finalImageList.push_back(transform_single_image(img_xform, img->get_image(i)));
    }

    auto rc = make_shared<image::decoded>();
    if(rc->add(finalImageList) == false) {
        rc = nullptr;
    }
    return rc;
}

cv::Mat image::transformer::transform_single_image(
                                            shared_ptr<image::params> img_xform,
                                            cv::Mat& single_img)
{
    cv::Mat rotatedImage;
    image::rotate(single_img, rotatedImage, img_xform->angle);
    cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

    cv::Mat resizedImage;
    image::resize(croppedImage, resizedImage, img_xform->output_size);
    photo.cbsjitter(resizedImage, img_xform->contrast, img_xform->brightness, img_xform->saturation, img_xform->hue);
    photo.lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

    cv::Mat *finalImage = &resizedImage;
    cv::Mat flippedImage;
    if (img_xform->flip) {
        cv::flip(resizedImage, flippedImage, 1);
        finalImage = &flippedImage;
    }
    return *finalImage;
}

shared_ptr<image::params>
image::param_factory::make_params(shared_ptr<const decoded> input)
{
    // Must use this method for creating a shared_ptr rather than make_shared
    // since the params default ctor is private and factory is friend
    // make_shared is not friend :(
    auto settings = shared_ptr<image::params>(new image::params());

    settings->output_size = cv::Size2i(_cfg.width, _cfg.height);

    settings->angle = _cfg.angle(_dre);
    settings->flip  = _cfg.flip_distribution(_dre);

    if(!_cfg.crop_enable)
    {
        cv::Size2f size = input->get_image_size();
        settings->cropbox = cv::Rect(cv::Point2f(0,0), size);
        float image_scale;
        if(_cfg.fixed_scaling_factor > 0) {
            image_scale = _cfg.fixed_scaling_factor;
        } else {
            image_scale = image::calculate_scale(size, _cfg.width, _cfg.height);
        }
//        settings->output_size = size * image_scale;
        size = size * image_scale;
        settings->output_size.width  = nervana::unbiased_round(size.width);
        settings->output_size.height = nervana::unbiased_round(size.height);
    }
    else
    {
        cv::Size2f in_size = input->get_image_size();

        float scale = _cfg.scale(_dre);
        float horizontal_distortion = _cfg.horizontal_distortion(_dre);
        cv::Size2f out_shape(_cfg.width * horizontal_distortion, _cfg.height);

        cv::Size2f cropbox_size = image::cropbox_max_proportional(in_size, out_shape);
        if(_cfg.do_area_scale) {
            cropbox_size = image::cropbox_area_scale(in_size, cropbox_size, scale);
        } else {
            cropbox_size = image::cropbox_linear_scale(cropbox_size, scale);
        }

        float c_off_x = _cfg.crop_offset(_dre);
        float c_off_y = _cfg.crop_offset(_dre);

        cv::Point2f cropbox_origin = image::cropbox_shift(in_size, cropbox_size, c_off_x, c_off_y);
        settings->cropbox = cv::Rect(cropbox_origin, cropbox_size);
    }

    if (_cfg.lighting.stddev() != 0) {
        for( int i=0; i<3; i++ ) {
            settings->lighting.push_back(_cfg.lighting(_dre));
        }
        settings->color_noise_std = _cfg.lighting.stddev();
    }

    return settings;
}

image::loader::loader(const image::config& cfg) :
    channel_major{cfg.channel_major},
    fixed_aspect_ratio{cfg.fixed_aspect_ratio},
    stype{cfg.get_shape_type()},
    channels{cfg.channels}
{
}

void image::loader::load(const std::vector<void*>& outlist, shared_ptr<image::decoded> input)
{
    char* outbuf = (char*)outlist[0];
    // TODO: Generalize this to also handle multi_crop case
    auto cv_type = stype.get_otype().cv_type;
    auto element_size = stype.get_otype().size;
    auto img = input->get_image(0);
    int image_size = img.channels() * img.total() * element_size;

    for (int i=0; i < input->get_image_count(); i++)
    {
        auto outbuf_i = outbuf + (i * image_size);
        auto input_image = input->get_image(i);
        vector<cv::Mat> source;
        vector<cv::Mat> target;
        vector<int>     from_to;

        if (fixed_aspect_ratio)
        {
            // zero out the output buffer as the image may not fill the canvas
            for(int i=0; i<stype.get_byte_size(); i++) outbuf[i] = 0;

            vector<size_t> shape = stype.get_shape();
            // methods for image_var
            if (channel_major)
            {
                // Split into separate channels
                int width    = shape[1];
                int height   = shape[2];
                int pix_per_channel = width * height;
                cv::Mat b(width, height, CV_8U, outbuf);
                cv::Mat g(width, height, CV_8U, outbuf + pix_per_channel);
                cv::Mat r(width, height, CV_8U, outbuf + 2 * pix_per_channel);
                cv::Rect roi(0, 0, input_image.cols, input_image.rows);
                cv::Mat b_roi = b(roi);
                cv::Mat g_roi = g(roi);
                cv::Mat r_roi = r(roi);
                cv::Mat channels[3] = {b_roi, g_roi, r_roi};
                cv::split(input_image, channels);
            }
            else
            {
                int channels = shape[2];
                int width    = shape[1];
                int height   = shape[0];
                cv::Mat output(height, width, CV_8UC(channels), outbuf);
                cv::Mat target_roi = output(cv::Rect(0, 0, input_image.cols, input_image.rows));
                input_image.copyTo(target_roi);
            }
        }
        else
        {
            // methods for image
            source.push_back(input_image);
            if (channel_major)
            {
                for(int ch=0; ch<channels; ch++)
                {
                    target.emplace_back(img.size(), cv_type, (char*)(outbuf_i + ch * img.total() * element_size));
                    from_to.push_back(ch);
                    from_to.push_back(ch);
                }
            }
            else
            {
                target.emplace_back(input_image.size(), CV_MAKETYPE(cv_type, channels), (char*)(outbuf_i));
                for(int ch=0; ch<channels; ch++)
                {
                    from_to.push_back(ch);
                    from_to.push_back(ch);
                }
            }
            image::convert_mix_channels(source, target, from_to);
        }
    }
}