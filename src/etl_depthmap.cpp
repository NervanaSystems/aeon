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

#include "etl_depthmap.hpp"

using namespace std;
using namespace nervana;

depthmap::extractor::extractor(const image::config& config)
{
}

depthmap::extractor::~extractor()
{
}

shared_ptr<image::decoded> depthmap::extractor::extract(const void* inbuf, size_t insize) const
{
    cv::Mat image;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, CV_8UC1, (char*)inbuf);
    cv::imdecode(input_img, CV_LOAD_IMAGE_ANYDEPTH, &image);

    // convert input image to single channel if needed
    if (image.channels() > 1)
    {
        // copy channel 0 from source image to channel 0 of target image where
        // target is a single channel image
        cv::Mat target(image.rows, image.cols, CV_8UC1);
        int     from_to[] = {0, 0};
        cv::mixChannels(&image, 1, &target, 1, from_to, 1);
        image = target;
    }

    return make_shared<image::decoded>(image);
}

depthmap::transformer::transformer(const image::config& config)
{
}

depthmap::transformer::~transformer()
{
}

std::shared_ptr<image::decoded>
    depthmap::transformer::transform(std::shared_ptr<augment::image::params> img_xform,
                                     std::shared_ptr<image::decoded>         image_list) const
{
    if (image_list->get_image_count() != 1)
        throw invalid_argument("depthmap transform only supports a single image");

    cv::Mat    rotatedImage;
    cv::Scalar border{0, 0, 0};
    image::rotate(image_list->get_image(0), rotatedImage, img_xform->angle, false, border);

    cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

    cv::Mat resizedImage;
    image::resize(croppedImage, resizedImage, img_xform->output_size, false);

    cv::Mat* finalImage = &resizedImage;
    cv::Mat  flippedImage;
    if (img_xform->flip)
    {
        cv::flip(resizedImage, flippedImage, 1);
        finalImage = &flippedImage;
    }

    return make_shared<image::decoded>(*finalImage);
}

void depthmap::loader::load(const std::vector<void*>&  outlist,
                            shared_ptr<image::decoded> input) const
{
    char* outbuf = (char*)outlist[0];
    // TODO: Generalize this to also handle multi_crop case
    auto img          = input->get_image(0);
    auto cv_type      = _cfg.get_shape_type().get_otype().get_cv_type();
    auto element_size = _cfg.get_shape_type().get_otype().get_size();
    int  image_size   = img.channels() * img.total() * element_size;

    for (int i = 0; i < input->get_image_count(); i++)
    {
        auto outbuf_i = outbuf + (i * image_size);
        img           = input->get_image(i);
        vector<cv::Mat> source;
        vector<cv::Mat> target;
        vector<int>     from_to;

        source.push_back(img);
        if (_cfg.channel_major)
        {
            for (int ch = 0; ch < _cfg.channels; ch++)
            {
                target.emplace_back(
                    img.size(), cv_type, (char*)(outbuf_i + ch * img.total() * element_size));
                from_to.push_back(ch);
                from_to.push_back(ch);
            }
        }
        else
        {
            target.emplace_back(img.size(), CV_MAKETYPE(cv_type, _cfg.channels), (char*)(outbuf_i));
            for (int ch = 0; ch < _cfg.channels; ch++)
            {
                from_to.push_back(ch);
                from_to.push_back(ch);
            }
        }
        image::convert_mix_channels(source, target, from_to);
    }
}
