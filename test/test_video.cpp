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

#include "gtest/gtest.h"

#include "etl_video.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

TEST(video, extract_transform)
{
    int    width      = 352;
    int    height     = 288;
    int    nseconds   = 1;
    int    frame_rate = 25;
    string test_file  = "output.avi";

    stringstream vidgen_command;
    vidgen_command << "ffmpeg -loglevel quiet -hide_banner ";
    vidgen_command << "-f lavfi -i testsrc=duration=" << nseconds;
    vidgen_command << ":size=" << width << "x" << height;
    vidgen_command << ":rate=" << frame_rate << " ";
    vidgen_command << "-c:v mjpeg -q:v 3 -y ";
    vidgen_command << test_file;

    auto a = system(vidgen_command.str().c_str());

    if (a == 0)
    {
        basic_ifstream<char> ifs(test_file, ios::binary);
        // read the data:
        vector<char> buf((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());

        // extract
        nlohmann::json js = {{"max_frame_count", 5},
                             {"frame", {{"height", height}, {"width", width}}}};

        video::config config(js);

        video::extractor extractor{config};
        auto             decoded_vid = extractor.extract((const char*)buf.data(), buf.size());

        ASSERT_EQ(decoded_vid->get_image_count(), 25);
        ASSERT_EQ(decoded_vid->get_image_size(), cv::Size2i(width, height));

        // transform
        video::transformer transformer = video::transformer(config);

        nlohmann::json                aug;
        augment::image::param_factory factory(aug);
        auto                          image_size = decoded_vid->get_image_size();
        auto                          params     = factory.make_params(
            image_size.width, image_size.height, config.frame.width, config.frame.height);

        params->output_size  = cv::Size2i(width / 2, height / 2);
        auto transformed_vid = transformer.transform(params, decoded_vid);
        ASSERT_NE(nullptr, transformed_vid);

        // make sure we've clipped the number of frames down according to max_frame_count
        // and that size has been reduced
        ASSERT_EQ(transformed_vid->get_image_count(), 5);
        ASSERT_EQ(transformed_vid->get_image_size(), cv::Size2i(width / 2, height / 2));

        remove(test_file.c_str());
    }
    else
    {
        ERR << "Missing ffmpeg for video extraction test" << endl;
    }
}

TEST(video, image_transform)
{
    int width  = 352;
    int height = 288;

    auto    decoded_image = make_shared<image::decoded>();
    cv::Mat mat_image(height, width, CV_8UC3, 0.0);
    decoded_image->add(mat_image);

    nlohmann::json js = {{"width", width}, {"height", height}};
    nlohmann::json aug;
    image::config  config(js);

    image::transformer _imageTransformer(config);

    augment::image::param_factory factory(aug);
    auto                          image_size = decoded_image->get_image_size();
    auto                          imageParams =
        factory.make_params(image_size.width, image_size.height, config.width, config.height);
    imageParams->output_size = cv::Size2i(width / 2, height / 2);

    _imageTransformer.transform(imageParams, decoded_image);
}

unsigned char expected_value(int d, int h, int w, int c)
{
    // set up expected_value in final outbuf so that viewed in
    // order in memory you see 0, 1, 2, 3, ...
    // the expected value of outbuf at index expected_value is expected_value
    return (((((c * 5) + d) * 4) + h) * 2) + w;
}

TEST(video, loader)
{
    // set up video::decoded with specific values
    // the color of any pixel channel should
    // = channel
    // + width * 3
    // + height * 2 * 3
    // + depth * 4 * 2 * 3
    // each dimension is unique to help debug and detect incorrect
    // dimension ordering
    // extract
    nlohmann::json js = {{"max_frame_count", 5},
                         {"frame", {{"channels", 3}, {"height", 4}, {"width", 2}}}};
    video::config vconfig{js};

    int channels, height, width, depth;
    tie(channels, height, width, depth) = make_tuple(
        vconfig.frame.channels, vconfig.frame.height, vconfig.frame.width, vconfig.max_frame_count);

    auto decoded_vid = make_shared<image::decoded>();

    for (int d = 0; d < depth; ++d)
    {
        cv::Mat image(height, width, CV_8UC3, 0.0);
        for (int w = 0; w < width; ++w)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int c = 0; c < channels; ++c)
                {
                    image.at<cv::Vec3b>(h, w).val[c] = expected_value(d, h, w, c);
                }
            }
        }
        decoded_vid->add(image);
    }

    // now run the loader
    vector<unsigned char> outbuf;

    int outbuf_size = channels * width * height * depth;
    outbuf.resize(outbuf_size);

    video::loader loader(vconfig);
    loader.load({outbuf.data()}, decoded_vid);

    // make sure outbuf has data in it like we expect
    for (int c = 0; c < channels; ++c)
    {
        for (int d = 0; d < depth; ++d)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    unsigned char v = expected_value(d, h, w, c);
                    ASSERT_EQ(outbuf[v], v);
                }
            }
        }
    }
}
