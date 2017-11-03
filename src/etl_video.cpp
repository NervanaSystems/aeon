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

#include "etl_video.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

std::shared_ptr<image::decoded> video::extractor::extract(const void* item, size_t itemSize) const
{
    // Very bad -- need to circle back and make an imemstream so we don't have to strip
    // constness from item
    char*                         bare_item = (char*)item;
    shared_ptr<MotionJpegCapture> mjdecoder = make_shared<MotionJpegCapture>(bare_item, itemSize);

    if (!mjdecoder->isOpened())
    {
        return nullptr;
    }
    auto    out_img = make_shared<image::decoded>();
    cv::Mat image;
    while (mjdecoder->grabFrame() && mjdecoder->retrieveFrame(0, image))
    {
        out_img->add(image.clone());
    }
    return out_img;
}

video::transformer::transformer(const video::config& config)
    : frame_transformer(config.frame)
    , max_frame_count(config.max_frame_count)
{
}

std::shared_ptr<image::decoded>
    video::transformer::transform(std::shared_ptr<augment::image::params> img_xform,
                                  std::shared_ptr<image::decoded>         img) const
{
    auto tx_img  = frame_transformer.transform(img_xform, img);
    auto out_img = make_shared<image::decoded>();

    uint32_t nframes = std::min<int>(max_frame_count, tx_img->get_image_count());

    for (uint32_t i = 0; i < nframes; ++i)
    {
        out_img->add(tx_img->get_image(i));
    }

    // Now pad out if necessary
    cv::Mat pad_frame = cv::Mat::zeros(img_xform->output_size, img->get_image(0).type());
    while (out_img->get_image_count() < max_frame_count)
    {
        if (out_img->add(pad_frame) == false)
        {
            out_img = nullptr;
        }
    }

    return out_img;
}

void video::loader::load(const vector<void*>& buflist, shared_ptr<image::decoded> input) const
{
    char* outbuf = (char*)buflist[0];
    // loads in channel x depth(frame) x height x width
    int        num_channels = input->get_image_channels();
    int        channel_size = input->get_image_count() * input->get_image_size().area();
    cv::Size2i image_size   = input->get_image_size();

    for (int i = 0; i < input->get_image_count(); i++)
    {
        auto img = input->get_image(i);

        auto image_offset = image_size.area() * i;

        if (num_channels == 1)
        {
            memcpy(outbuf + image_offset, img.data, image_size.area());
        }
        else
        {
            // create views into outbuf for the 3 channels to be copied into
            cv::Mat b(image_size, CV_8U, outbuf + 0 * channel_size + image_offset);
            cv::Mat g(image_size, CV_8U, outbuf + 1 * channel_size + image_offset);
            cv::Mat r(image_size, CV_8U, outbuf + 2 * channel_size + image_offset);

            cv::Mat channels[3] = {b, g, r};
            // cv::split will split img into component channels and copy
            // them into the addresses at b, g, r
            cv::split(img, channels);
        }
    }
}
