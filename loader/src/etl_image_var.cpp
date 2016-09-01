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

#include "etl_image_var.hpp"
#include "etl_localization.hpp"

using namespace std;
using namespace nervana;

image_var::config::config(nlohmann::json js)
{
    if(js.is_null()) {
        throw std::runtime_error("missing image_var config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config("image_var", config_list, js);

    // Now fill in derived
    shape_t shape;
    if (flip_enable) {
        flip_distribution = bernoulli_distribution{0.5};
    }

    if (!center) {
        crop_offset = uniform_real_distribution<float> {0.0f, 1.0f};
    }

    if (channel_major) {
        shape = {channels, max_size, max_size};
    } else{
        shape = {max_size, max_size, channels};
    }
    add_shape_type(shape, output_type);

    validate();
}

void image_var::config::validate() {
    if(max_size < min_size) {
        throw std::invalid_argument("max_size must be greater than or equal to min_size");
    }
}

void image_var::params::dump(ostream & ostr)
{
    ostr << "Flip: " << flip << "\n";
}


/* Extract */
image_var::extractor::extractor(const image_var::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw std::runtime_error(ss.str());
    } else {
        pixel_type = cfg.channels == 1 ? CV_8UC1 : CV_8UC3;
        color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

shared_ptr<image_var::decoded> image_var::extractor::extract(const char* inbuf, int insize)
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, pixel_type, const_cast<char*>(inbuf));
    cv::imdecode(input_img, color_mode, &output_img);

    return make_shared<image_var::decoded>(output_img);
}


/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly determined flip (based on params->flip)

*/

image_var::transformer::transformer(const image_var::config& cfg) :
    min_size{cfg.min_size},
    max_size{cfg.max_size}
{
}

shared_ptr<image_var::decoded> image_var::transformer::transform(
                                                 shared_ptr<image_var::params> img_xform,
                                                 shared_ptr<image_var::decoded> img)
{
    cv::Mat image = img->get_image();
    cv::Mat output;
    cv::Size im_size;
    float im_scale;
    im_scale = image::calculate_scale(image.size(), min_size, max_size);
    im_size = cv::Size{int(unbiased_round(image.size().width*im_scale)), int(unbiased_round(image.size().height*im_scale))};

    nervana::image::resize(image, output, im_size);

    if (img_xform->flip) {
        cv::Mat flippedImage;
        cv::flip(output, flippedImage, 1);
        output = flippedImage;
    }

    auto rc = make_shared<image_var::decoded>(output);
    return rc;
}

shared_ptr<image_var::params>
image_var::param_factory::make_params(shared_ptr<const decoded> input)
{
    // Must use this method for creating a shared_ptr rather than make_shared
    // since the params default ctor is private and factory is friend
    // make_shared is not friend :(
    auto params = shared_ptr<image_var::params>(new image_var::params());

    params->flip  = config.flip_distribution(generator);
    params->output_size = input->get_image_size();
    params->cropbox = cv::Rect(cv::Point(0,0), input->get_image_size());

    return params;
}

image_var::loader::loader(const image_var::config& cfg) :
    channel_major{cfg.channel_major},
    stype{cfg.get_shape_type()}
{
}

void image_var::loader::load(const vector<void*>& outlist, shared_ptr<image_var::decoded> input)
{
    char* outbuf = (char*)outlist[0];
    auto img = input->get_image();
    vector<size_t> shape = stype.get_shape();
    int output_buffer_size = stype.get_byte_size();
    for(int i=0; i<output_buffer_size; i++) outbuf[i] = 0;
    cv::Mat input_image = input->get_image();

    if (channel_major) {
        // Split into separate channels
        int width  = shape[1];
        int height = shape[2];
        int pix_per_channel = width * height;
        cv::Mat b(width, height, CV_8U, outbuf);
        cv::Mat g(width, height, CV_8U, outbuf + pix_per_channel);
        cv::Mat r(width, height, CV_8U, outbuf + 2 * pix_per_channel);
        cv::Rect roi(0, 0, input_image.cols, input_image.rows);
        cv::Mat b_roi = b(roi);
        cv::Mat g_roi = g(roi);
        cv::Mat r_roi = r(roi);
        cv::Mat channels[3] = {b_roi, g_roi, r_roi};
        cv::split(img, channels);
    } else {
        cv::Mat output(shape[0], shape[1], CV_8UC(shape[2]), outbuf);
        cv::Mat target_roi = output(cv::Rect(0, 0, input_image.cols, input_image.rows));
        input_image.copyTo(target_roi);
    }
}
