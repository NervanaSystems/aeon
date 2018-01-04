/*
 Copyright 2017 Nervana Systems Inc.
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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gtest/gtest.h"

#define private public

#include "etl_depthmap.hpp"
#include "json.hpp"
#include "helpers.hpp"

using namespace std;
using namespace nervana;

cv::Mat generate_test_image()
{
    cv::Mat        color = cv::Mat(256, 256, CV_8UC3);
    unsigned char* input = (unsigned char*)(color.data);
    int            index = 0;
    for (int row = 0; row < 256; row++)
    {
        for (int col = 0; col < 256; col++)
        {
            uint8_t value  = ((row + col) % 2 ? 0xFF : 0x00);
            input[index++] = value; // b
            input[index++] = value; // g
            input[index++] = value; // r
        }
    }
    return color;
}

// pixels must be either black or white
bool verify_image(cv::Mat img)
{
    unsigned char* data  = (unsigned char*)(img.data);
    int            index = 0;
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            for (int channel = 0; channel < img.channels(); channel++)
            {
                if (data[index] != 0x00 && data[index] != 0xFF)
                    return false;
                index++;
            }
        }
    }
    return true;
}

#ifdef PYTHON_PLUGIN
TEST(plugin, depthmap_example_rotate)
{
    auto            test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);
    nlohmann::json plugin_params;
    plugin_params["angle"] = {45, 45};

    depthmap::extractor           extractor{cfg};
    depthmap::transformer         transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    auto params_ptr =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params_ptr->user_plugin = make_shared<nervana::plugin>("rotate", plugin_params.dump());
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                    tximg       = transformed->get_image(0);
    cv::imwrite("tx_depthmap_rotate_plugin.png", tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(plugin, depthmap_example_flip)
{
    auto            test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);

    nlohmann::json js  = {{"width", 256}, {"height", 256}};
    nlohmann::json aug = {{"type", "image"},
                          {"crop_enable", false},
                          {"plugin_filename", "flip"},
                          {"plugin_params", {{"probability", 1}, {"width", 256}}}};
    image::config cfg(js);

    depthmap::extractor           extractor{cfg};
    depthmap::transformer         transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    auto params_ptr =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                    tximg       = transformed->get_image(0);

    // phase two
    aug = {{"type", "image"}, {"crop_enable", false}, {"flip_enable", true}};
    augment::image::param_factory factory2{aug};
    params_ptr = factory2.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params_ptr->flip = true;
    transformed      = transformer.transform(params_ptr, extracted);
    cv::Mat tximg2   = transformed->get_image(0);

    // compare
    bool isEqual = (cv::sum(tximg != tximg2) == cv::Scalar(0, 0, 0, 0));
    EXPECT_TRUE(isEqual);
}
#endif
