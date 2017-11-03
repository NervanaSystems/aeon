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

#include <vector>
#include <string>
#include <sstream>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#define private public

#include "etl_pixel_mask.hpp"
#include "json.hpp"
#include "helpers.hpp"
#include "provider_factory.hpp"

using namespace std;
using namespace nervana;

static cv::Mat generate_test_image()
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
static bool verify_image(cv::Mat img)
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

TEST(pixel_mask, scale_up)
{
    auto            test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    pixel_mask::extractor         extractor{cfg};
    pixel_mask::transformer       transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr  = builder.output_size(300, 300);
    shared_ptr<image::decoded>         transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                            tximg       = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_scale_up.png", tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, scale_down)
{
    auto            test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    pixel_mask::extractor         extractor{cfg};
    pixel_mask::transformer       transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr  = builder.output_size(100, 100);
    shared_ptr<image::decoded>         transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                            tximg       = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_scale_down.png", tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, rotate)
{
    auto            test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    pixel_mask::extractor         extractor{cfg};
    pixel_mask::transformer       transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr  = builder.angle(45);
    shared_ptr<image::decoded>         transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                            tximg       = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_rotate.png", tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, load_int)
{
    cv::Mat  test_image(256, 256, CV_8UC3);
    uint8_t* input = (uint8_t*)(test_image.data);
    int      index = 0;
    for (int row = 0; row < 256; row++)
    {
        for (int col = 0; col < 256; col++)
        {
            uint8_t value  = col;
            input[index++] = value; // b
            input[index++] = value; // g
            input[index++] = value; // r
        }
    }

    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);

    nlohmann::json js = {
        {"width", 256}, {"height", 256}, {"output_type", "int32_t"}, {"channels", 1}};
    nlohmann::json aug;
    image::config  cfg(js);

    pixel_mask::extractor         extractor{cfg};
    pixel_mask::transformer       transformer{cfg};
    image::loader                 loader{cfg, false};
    augment::image::param_factory factory{aug};

    auto extracted  = extractor.extract((const char*)test_data.data(), test_data.size());
    auto image_size = extracted->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr  = builder;
    shared_ptr<image::decoded>         transformed = transformer.transform(params_ptr, extracted);
    cv::Mat                            tximg       = transformed->get_image(0);

    cv::Mat output_image(256, 256, CV_32SC1);
    loader.load({output_image.data}, transformed);

    {
        uint8_t* output = output_image.data;
        index           = 0;
        for (int row = 0; row < 256; row++)
        {
            for (int col = 0; col < 256; col++)
            {
                uint8_t value = col;
                ASSERT_EQ(value, unpack<int32_t>(&output[sizeof(int32_t) * index++]));
            }
        }
    }
}

// TEST(pixel_mask, provider)
//{
//    string image_path = CURDIR"/test_data/segnet/image/0001TP_006690.png";
//    string annot_path = CURDIR"/test_data/segnet/annot/0001TP_006690.png";

//    nlohmann::json js = {{"type","image,pixelmask"},
//                         {"image", {
//                            {"height",256},
//                            {"width",512},
//                            {"channel_major",false}
//                            }},
//                         {"pixelmask", {
//                            {"height",256},
//                            {"width",512},
//                            {"channel_major",false},
//                            {"channels", 1}
//                            }}
//                        };

//    auto media = nervana::train_provider_factory::create(js);
//    ASSERT_NE(nullptr, media);

//    auto image_data = read_file_contents(image_path);
//    auto annot_data = read_file_contents(annot_path);
//    ASSERT_TRUE(image_data.size()>0) << image_path;
//    ASSERT_TRUE(annot_data.size()>0) << annot_path;

//    size_t batch_size = 4;

//    buffer_in_array bp(2);
//    buffer_in& data_p = *bp[0];
//    buffer_in& target_p = *bp[1];

//    for (int i=0; i<batch_size; i++) {
//        data_p.add_item(image_data);
//        target_p.add_item(annot_data);
//    }

//    EXPECT_EQ(data_p.record_count(), batch_size);
//    EXPECT_EQ(target_p.record_count(), batch_size);

//    ASSERT_EQ(2, media->get_output_shapes().size());

//    // Generate output buffers using shapes from the provider
//    buffer_out_array outBuf({media->get_output_shapes()[0].get_byte_size(),
//                             media->get_output_shapes()[1].get_byte_size()},
//                            batch_size);

//    // Call the provider
//    for (int i=0; i<batch_size; i++)
//    {
//        media->provide(i, bp, outBuf);
//    }

////    unsigned char *data = (unsigned char*)(tximg.data);
////    int index = 0;
////    for(int row = 0; row < tximg.rows; row++) {
////        for(int col = 0; col < tximg.cols; col++) {
////            int b = data[index++];
////            int g = data[index++];
////            int r = data[index++];
////            cout << b << "," << g << "," << r << endl;
////        }
////    }
//}
