/*
 Copyright 2016-2017 Intel(R) Nervana(TM)
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

#include "json.hpp"
#include "helpers.hpp"
#include "image.hpp"
#include "log.hpp"
#include "util.hpp"
#include "file_util.hpp"

#define private public

#include "etl_image.hpp"

using namespace std;
using namespace nervana;

static cv::Mat generate_indexed_image(int rows, int cols)
{
    cv::Mat        color = cv::Mat(rows, cols, CV_8UC3);
    unsigned char* input = (unsigned char*)(color.data);
    int            index = 0;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            input[index++] = col; // b
            input[index++] = row; // g
            input[index++] = 0;   // r
        }
    }
    return color;
}

static cv::Mat generate_stripped_image(int          width,
                                       int          height,
                                       unsigned int left_color,
                                       unsigned int right_color)
{
    cv::Mat  mat  = cv::Mat(height, width, CV_8UC3);
    uint8_t* data = mat.data;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width / 2; j++)
        {
            data[0] = left_color & 0xFF;
            data[1] = ((left_color & 0xFF00) >> 8);
            data[2] = ((left_color & 0xFF0000) >> 16);
            data += 3;
        }
        for (int j = 0; j < width / 2; j++)
        {
            data[0] = right_color & 0xFF;
            data[1] = ((right_color & 0xFF00) >> 8);
            data[2] = ((right_color & 0xFF0000) >> 16);
            data += 3;
        }
    }
    return mat;
}

static void test_image(vector<unsigned char>& img, int channels)
{
    nlohmann::json js = {{"channels", channels}, {"height", 30}, {"width", 30}};

    image::config itpj(js);

    image::extractor           ext{itpj};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    ASSERT_NE(nullptr, decoded);
    EXPECT_EQ(1, decoded->get_image_count());
    cv::Size2i size = decoded->get_image_size();
    EXPECT_EQ(256, size.width);
    EXPECT_EQ(256, size.height);
    cv::Mat mat = decoded->get_image(0);
    EXPECT_EQ(256, mat.rows);
    EXPECT_EQ(256, mat.cols);
    EXPECT_EQ(channels, mat.channels());

    // unsigned char *input = (unsigned char*)(mat.data);
    // int index = 0;
    // for(int row = 0; row < 256; row++) {
    //     for(int col = 0; col < 256; col++) {
    //         if(channels == 3) {
    //             EXPECT_EQ(col,input[index++]);
    //             EXPECT_EQ(row,input[index++]);
    //             index++;
    //         } else if(channels == 1) {
    //         }
    //     }
    // }
}

TEST(image, passthrough)
{
    cv::Mat        test_image = cv::Mat(256, 512, CV_8UC3);
    unsigned char* input      = (unsigned char*)(test_image.data);
    int            index      = 0;
    for (int row = 0; row < test_image.rows; row++)
    {
        for (int col = 0; col < test_image.cols; col++)
        {
            input[index++] = col; // b
            input[index++] = row; // g
            input[index++] = 0;   // r
        }
    }

    vector<unsigned char> image_data;
    cv::imencode(".png", test_image, image_data);

    nlohmann::json js = {{"width", 512}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&image_data[0], image_data.size());

    augment::image::param_factory factory(aug);

    auto image_size = decoded->get_image_size();
    auto params_ptr =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);

    image::transformer         trans{cfg};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);

    cv::imwrite("size_input_image.png", test_image);
    cv::imwrite("size_output_image.png", image);

    unsigned char* input_data  = (unsigned char*)(test_image.data);
    unsigned char* output_data = (unsigned char*)(image.data);

    for (int i = 0; i < test_image.rows * test_image.cols * 3; i++)
    {
        ASSERT_EQ(input_data[i], output_data[i]);
    }

    //    EXPECT_EQ(20,image.size().width);
    //    EXPECT_EQ(30,image.size().height);

    //    EXPECT_TRUE(check_value(transformed,0,0,100,150));
    //    EXPECT_TRUE(check_value(transformed,19,0,119,150));
    //    EXPECT_TRUE(check_value(transformed,0,29,100,179));
}

TEST(image, decoded)
{
    cv::Mat img1 = cv::Mat(256, 256, CV_8UC3);
    cv::Mat img2 = cv::Mat(256, 256, CV_8UC3);
    cv::Mat img3 = cv::Mat(256, 256, CV_8UC3);
    cv::Mat img4 = cv::Mat(100, 100, CV_8UC3);

    vector<cv::Mat> v1{img1, img2, img3};
    vector<cv::Mat> v2{img4};

    image::decoded decoded;
    EXPECT_TRUE(decoded.add(img1));
    EXPECT_TRUE(decoded.add(img2));
    EXPECT_TRUE(decoded.add(img3));
    EXPECT_TRUE(decoded.add(v1));
    EXPECT_FALSE(decoded.add(img4)); // image size does not match
    EXPECT_FALSE(decoded.add(v2));
}

TEST(image, missing_config_arg)
{
    nlohmann::json js = {{"width", 30},
                         {"channels", 1},
                         {"angle", {-20, 20}},
                         {"scale", {0.2, 0.8}},
                         {"lighting", {0.0, 0.1}},
                         {"horizontal_distortion", {0.75, 1.33}},
                         {"flip_enable", false}};

    EXPECT_THROW(image::config itpj(js), std::invalid_argument);
}

TEST(image, config)
{
    nlohmann::json js = {{"height", 30}, {"width", 30}, {"channels", 3}};

    image::config config(js);
    EXPECT_EQ(30, config.height);
    EXPECT_EQ(30, config.width);
    EXPECT_TRUE(config.channel_major);
    EXPECT_EQ(3, config.channels);
}

TEST(image, extract1)
{
    auto                  indexed = generate_indexed_image(256, 256);
    vector<unsigned char> png;
    cv::imencode(".png", indexed, png);

    test_image(png, 3);
}

TEST(image, extract2)
{
    auto                  indexed = generate_indexed_image(256, 256);
    vector<unsigned char> png;
    cv::imencode(".png", indexed, png);

    test_image(png, 1);
}

TEST(image, extract3)
{
    cv::Mat               img = cv::Mat(256, 256, CV_8UC1, 0.0);
    vector<unsigned char> png;
    cv::imencode(".png", img, png);

    test_image(png, 3);
}

TEST(image, extract4)
{
    cv::Mat               img = cv::Mat(256, 256, CV_8UC1, 0.0);
    vector<unsigned char> png;
    cv::imencode(".png", img, png);

    test_image(png, 1);
}

bool check_value(shared_ptr<image::decoded> transformed, int x0, int y0, int x1, int y1, int ii = 0)
{
    cv::Mat   image = transformed->get_image(ii);
    cv::Vec3b value = image.at<cv::Vec3b>(y0, x0); // row,col
    return x1 == (int)value[0] && y1 == (int)value[1];
}

TEST(image, transform_crop)
{
    auto                  indexed = generate_indexed_image(256, 256);
    vector<unsigned char> img;
    cv::imencode(".png", indexed, img);

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    augment::image::param_factory factory(aug);

    auto                 image_size = decoded->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr =
        builder.cropbox(100, 150, 20, 30).output_size(20, 30);

    image::transformer         trans{cfg};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(20, image.size().width);
    EXPECT_EQ(30, image.size().height);

    EXPECT_TRUE(check_value(transformed, 0, 0, 100, 150));
    EXPECT_TRUE(check_value(transformed, 19, 0, 119, 150));
    EXPECT_TRUE(check_value(transformed, 0, 29, 100, 179));
}

TEST(image, transform_expand_crop_flip_resize)
{
    int input_width   = 25;
    int input_height  = 25;
    int output_width  = 100;
    int output_height = 100;

    cv::Size2i expand_offset(50, 0);
    cv::Size2i expand_size(100, 100);
    float      expand_ratio = 4.0;

    cv::Rect cropbox = cv::Rect(50, 0, 50, 50);

    cv::Mat mat = cv::Mat(input_height, input_width, CV_8UC3);
    mat         = cv::Scalar(0xFF, 0, 0);

    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json js = {{"width", input_width}, {"height", input_height}, {"channels", 3}};
    nlohmann::json aug;
    image::config  cfg(js);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    augment::image::param_factory factory(aug);

    auto                 image_size = decoded->get_image_size();
    image_params_builder builder(factory.make_ssd_params(image_size.width,
                                                         image_size.height,
                                                         output_width,
                                                         output_height,
                                                         vector<boundingbox::box>()));
    shared_ptr<augment::image::params> params_ptr =
        builder.expand(expand_ratio, expand_offset, expand_size)
            .cropbox(cropbox)
            .flip(true)
            .output_size(output_width, output_height);

    image::transformer         trans{cfg};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(output_width, image.size().width);
    EXPECT_EQ(output_height, image.size().height);

    const int blurred_bias = 5;
    for (int i = 0; i < image.size().height; i++)
    {
        for (int j = 0; j < image.size().width; j++)
        {
            auto pixel = image.at<cv::Vec3b>(i, j);
            if (j >= output_width / 2 + blurred_bias && i < output_height / 2 - blurred_bias)
            {
                EXPECT_EQ(pixel[0], 0xFF);
                EXPECT_EQ(pixel[1], 0x0);
                EXPECT_EQ(pixel[2], 0x0);
            }
            else if (j < output_width / 2 - blurred_bias || i >= output_height / 2 + blurred_bias)
            {
                EXPECT_EQ(pixel[0], 0x0);
                EXPECT_EQ(pixel[1], 0x0);
                EXPECT_EQ(pixel[2], 0x0);
            }
        }
    }
}

TEST(image, transform_flip)
{
    auto                  indexed = generate_indexed_image(256, 256);
    vector<unsigned char> img;
    cv::imencode(".png", indexed, img);

    nlohmann::json js = {{"width", 256}, {"height", 256}};
    nlohmann::json aug;
    image::config  cfg(js);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    augment::image::param_factory factory(aug);

    auto                 image_size = decoded->get_image_size();
    image_params_builder builder(
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
    shared_ptr<augment::image::params> params_ptr =
        builder.cropbox(100, 150, 20, 20).output_size(20, 20).flip(true);

    image::transformer         trans{cfg};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(20, image.size().width);
    EXPECT_EQ(20, image.size().height);

    EXPECT_TRUE(check_value(transformed, 0, 0, 119, 150));
    EXPECT_TRUE(check_value(transformed, 19, 0, 100, 150));
    EXPECT_TRUE(check_value(transformed, 0, 19, 119, 169));
}

TEST(image, transform_padding)
{
    struct test
    {
        const int width;
        const int height;
        const int padding;
        const int crop_offset_x;
        const int crop_offset_y;
    };
    vector<test> tests = {{32, 32, 5, 2, 8},
                          {32, 32, 5, 5, 5},
                          {32, 32, 4, 0, 0},
                          {30, 30, 5, 10, 10},
                          {30, 30, 0, 0, 0},
                          {1, 1, 10, 0, 20},
                          {2, 2, 10, 10, 10}};

    for (const test& tc : tests)
    {
        auto                  indexed = generate_indexed_image(tc.height, tc.width);
        vector<unsigned char> img;
        cv::imencode(".png", indexed, img);

        nlohmann::json js = {{"width", tc.width}, {"height", tc.height}};
        nlohmann::json aug;
        image::config  cfg(js);

        image::extractor           ext{cfg};
        shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

        augment::image::param_factory factory(aug);

        auto                 image_size = decoded->get_image_size();
        image_params_builder builder(
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height));
        shared_ptr<augment::image::params> params_ptr =
            builder.output_size(tc.width, tc.height)
                .padding(tc.padding, tc.crop_offset_x, tc.crop_offset_y);

        image::transformer         trans{cfg};
        shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

        cv::Mat image = transformed->get_image(0);

        ASSERT_EQ(tc.width, image.size().width);
        ASSERT_EQ(tc.height, image.size().height);

        for (int row = 0; row < tc.height + tc.padding * 2; row++)
        {
            for (int col = 0; col < tc.width + tc.padding * 2; col++)
            {
                // out of cropped box
                if (row < tc.crop_offset_y || col < tc.crop_offset_x ||
                    row >= tc.crop_offset_y + tc.height || col >= tc.crop_offset_x + tc.width)
                    continue;

                auto pixel = image.at<cv::Vec3b>(row - tc.crop_offset_y, col - tc.crop_offset_x);
                // col = b row = g
                int r, g, b;
                // we are at padded black pixel
                if (col < tc.padding || col >= tc.width + tc.padding || row < tc.padding ||
                    row >= tc.height + tc.padding)
                {
                    r = g = b = 0;
                    EXPECT_EQ(b, pixel[0]) << "blue pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                    EXPECT_EQ(g, pixel[1]) << "green pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                    EXPECT_EQ(r, pixel[2]) << "red pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                }
                else // input image
                {
                    b = col - tc.padding;
                    g = row - tc.padding;
                    r = 0;
                    EXPECT_EQ(b, pixel[0]) << "blue pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                    EXPECT_EQ(g, pixel[1]) << "green pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                    EXPECT_EQ(r, pixel[2]) << "red pixel at row " << row << " col " << col
                                           << " is not equal to reference value";
                }
            }
        }
    }
}

TEST(image, noconvert_nosplit)
{
    nlohmann::json js = {{"width", 10},
                         {"height", 10},
                         {"channels", 3},
                         {"channel_major", false},
                         {"output_type", "uint8_t"}};
    image::config cfg(js);

    cv::Mat input_image(100, 100, CV_8UC3);
    input_image = cv::Scalar(50, 100, 200);
    cv::Mat output_image(100, 100, CV_8UC3);

    vector<unsigned char> image_data;
    cv::imencode(".png", input_image, image_data);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&image_data[0], image_data.size());

    image::loader loader(cfg, false);
    loader.load({output_image.data}, decoded);

    //    cv::imwrite("image_noconvert_nosplit.png", output_image);
    uint8_t* input = (uint8_t*)(output_image.data);
    int      index = 0;
    for (int row = 0; row < output_image.rows; row++)
    {
        for (int col = 0; col < output_image.cols; col++)
        {
            ASSERT_EQ(50, input[index++]);  // b
            ASSERT_EQ(100, input[index++]); // g
            ASSERT_EQ(200, input[index++]); // r
        }
    }
}

TEST(image, noconvert_split)
{
    nlohmann::json js = {{"width", 10},
                         {"height", 10},
                         {"channels", 3},
                         {"channel_major", true},
                         {"output_type", "uint8_t"}};
    image::config cfg(js);

    cv::Mat input_image(100, 100, CV_8UC3);
    input_image = cv::Scalar(50, 100, 150);
    cv::Mat output_image(300, 100, CV_8UC1);

    vector<unsigned char> image_data;
    cv::imencode(".png", input_image, image_data);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&image_data[0], image_data.size());

    image::loader loader(cfg, false);
    loader.load({output_image.data}, decoded);

    // cv::imwrite("image_noconvert_split.png", output_image);
    uint8_t* input = (uint8_t*)(output_image.data);
    int      index = 0;
    for (int ch = 0; ch < 3; ch++)
    {
        for (int row = 0; row < input_image.rows; row++)
        {
            for (int col = 0; col < input_image.cols; col++)
            {
                ASSERT_EQ(50 * (ch + 1), input[index++]);
            }
        }
    }
}

TEST(image, convert_nosplit)
{
    nlohmann::json js = {{"width", 10},
                         {"height", 10},
                         {"channels", 3},
                         {"channel_major", false},
                         {"output_type", "uint32_t"}};
    image::config cfg(js);

    cv::Mat input_image(100, 100, CV_8UC3);
    input_image = cv::Scalar(50, 100, 200);
    cv::Mat output_image(100, 100, CV_32SC3);

    vector<unsigned char> image_data;
    cv::imencode(".png", input_image, image_data);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&image_data[0], image_data.size());

    image::loader loader(cfg, false);
    loader.load({output_image.data}, decoded);

    //    cv::imwrite("image_convert_nosplit.png", output_image);
    int index = 0;
    for (int row = 0; row < output_image.rows; row++)
    {
        for (int col = 0; col < output_image.cols; col++)
        {
            ASSERT_EQ(50, unpack<int32_t>(output_image.data, sizeof(int32_t) * index++));  // b
            ASSERT_EQ(100, unpack<int32_t>(output_image.data, sizeof(int32_t) * index++)); // g
            ASSERT_EQ(200, unpack<int32_t>(output_image.data, sizeof(int32_t) * index++)); // r
        }
    }
}

TEST(image, convert_split)
{
    nlohmann::json js = {{"width", 10},
                         {"height", 10},
                         {"channels", 3},
                         {"channel_major", true},
                         {"output_type", "uint32_t"}};
    image::config cfg(js);

    cv::Mat input_image(100, 100, CV_8UC3);
    input_image = cv::Scalar(50, 100, 150);
    cv::Mat output_image(300, 100, CV_32SC1);

    vector<unsigned char> image_data;
    cv::imencode(".png", input_image, image_data);

    image::extractor           ext{cfg};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&image_data[0], image_data.size());

    image::loader loader(cfg, false);
    loader.load({output_image.data}, decoded);

    //    cv::imwrite("image_convert_split.png", output_image);
    int index = 0;
    for (int ch = 0; ch < 3; ch++)
    {
        for (int row = 0; row < input_image.rows; row++)
        {
            for (int col = 0; col < input_image.cols; col++)
            {
                ASSERT_EQ(50 * (ch + 1),
                          unpack<int32_t>(output_image.data, sizeof(int32_t) * index++));
            }
        }
    }
}

TEST(image, cropbox_max_proportional)
{
    {
        cv::Size2f in(100, 50);
        cv::Size2f out(200, 100);
        cv::Size2f result = image::cropbox_max_proportional(in, out);
        EXPECT_EQ(100, result.width);
        EXPECT_EQ(50, result.height);
    }

    {
        cv::Size2f in(100, 50);
        cv::Size2f out(50, 25);
        cv::Size2f result = image::cropbox_max_proportional(in, out);
        EXPECT_EQ(100, result.width);
        EXPECT_EQ(50, result.height);
    }

    {
        cv::Size2f in(100, 50);
        cv::Size2f out(200, 50);
        cv::Size2f result = image::cropbox_max_proportional(in, out);
        EXPECT_EQ(100, result.width);
        EXPECT_EQ(25, result.height);
    }

    {
        cv::Size2f in(100, 50);
        cv::Size2f out(50, 100);
        cv::Size2f result = image::cropbox_max_proportional(in, out);
        EXPECT_EQ(25, result.width);
        EXPECT_EQ(50, result.height);
    }

    {
        cv::Size2f in(100, 50);
        cv::Size2f out(10, 10);
        cv::Size2f result = image::cropbox_max_proportional(in, out);
        EXPECT_EQ(50, result.width);
        EXPECT_EQ(50, result.height);
    }
}

TEST(image, calculate_scale)
{
    int      width  = 800;
    int      height = 800;
    cv::Size size{500, 375};
    float    scale;
    scale = image::calculate_scale(size, width, height);
    size =
        cv::Size{int(unbiased_round(size.width * scale)), int(unbiased_round(size.height * scale))};
    EXPECT_FLOAT_EQ(1.6, scale);
    EXPECT_EQ(800, size.width);
    EXPECT_EQ(600, size.height);
}

TEST(image, transform)
{
    vector<char> image_data = file_util::read_file_contents(CURDIR "/test_data/img_2112_70.jpg");
    //        vector<char> image_data = read_file_contents(CURDIR"/test_data/test_image.jpg");
    {
        int            height   = 128;
        int            width    = 256;
        int            channels = 3;
        nlohmann::json js       = {
            {"height", height}, {"width", width}, {"channels", channels}, {"channel_major", false}};
        nlohmann::json aug = {{"type", "image"}, {"flip_enable", false}};

        image::config                 cfg{js};
        image::extractor              extractor{cfg};
        image::transformer            transformer{cfg};
        image::loader                 loader{cfg, false};
        augment::image::param_factory factory(aug);

        auto decoded    = extractor.extract(image_data.data(), image_data.size());
        auto image_size = decoded->get_image_size();
        auto params =
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
        auto transformed = transformer.transform(params, decoded);

        cv::Mat output_image(height, width, CV_8UC(channels));
        loader.load({output_image.data}, transformed);
        string filename = "image_transform_1.png";
        cv::imwrite(filename, output_image);
    }
    {
        int            height   = 128;
        int            width    = 256;
        int            channels = 3;
        nlohmann::json js       = {
            {"height", height}, {"width", width}, {"channels", channels}, {"channel_major", false}};
        nlohmann::json aug = {{"type", "image"}, {"flip_enable", false}};

        image::config                 cfg{js};
        image::extractor              extractor{cfg};
        image::transformer            transformer{cfg};
        image::loader                 loader{cfg, false};
        augment::image::param_factory factory(aug);

        auto decoded = extractor.extract(image_data.data(), image_data.size());

        auto                               image_size = decoded->get_image_size();
        shared_ptr<augment::image::params> params =
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
        params->flip = true;

        auto transformed = transformer.transform(params, decoded);

        cv::Mat output_image(height, width, CV_8UC(channels));
        loader.load({output_image.data}, transformed);
        string filename = "image_transform_2.png";
        cv::imwrite(filename, output_image);
    }
    {
        int            height   = 128;
        int            width    = 256;
        int            channels = 3;
        nlohmann::json js       = {
            {"height", height}, {"width", width}, {"channels", channels}, {"channel_major", false}};
        nlohmann::json aug = {{"type", "image"},
                              {"horizontal_distortion", {2, 2}},
                              {"scale", {0.5, 0.5}},
                              {"flip_enable", false}};

        image::config                 cfg{js};
        image::extractor              extractor{cfg};
        image::transformer            transformer{cfg};
        image::loader                 loader{cfg, false};
        augment::image::param_factory factory(aug);

        auto decoded = extractor.extract(image_data.data(), image_data.size());

        auto                               image_size = decoded->get_image_size();
        shared_ptr<augment::image::params> params =
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
        params->flip = false;

        auto transformed = transformer.transform(params, decoded);

        cv::Mat output_image(height, width, CV_8UC(channels));
        loader.load({output_image.data}, transformed);
        string filename = "image_transform_3.png";
        cv::imwrite(filename, output_image);
    }
}

TEST(image, config_bad_scale)
{
    int            height   = 128;
    int            width    = 256;
    int            channels = 3;
    nlohmann::json js       = {{"type", "image"},
                         {"height", height},
                         {"width", width},
                         {"channels", channels},
                         {"horizontal_distortion", {2, 2}},
                         {"scale", {0.5, 1.5}},
                         {"channel_major", false},
                         {"flip_enable", false}};

    EXPECT_THROW(image::config{js}, std::invalid_argument);
}

TEST(image, area_scale)
{
    vector<char> image_data = file_util::read_file_contents(CURDIR "/test_data/img_2112_70.jpg");
    float        max_cropbox_area;
    float        max_cropbox_ratio;
    float        source_image_area;

    {
        int            height   = 128;
        int            width    = 256;
        int            channels = 3;
        nlohmann::json js       = {
            {"height", height}, {"width", width}, {"channels", channels}, {"channel_major", false}};
        nlohmann::json aug = {{"type", "image"}, {"do_area_scale", true}, {"flip_enable", false}};

        {
            image::config                 cfg{js};
            image::extractor              extractor{cfg};
            augment::image::param_factory factory(aug);

            auto decoded      = extractor.extract(image_data.data(), image_data.size());
            source_image_area = decoded->get_image_size().area();

            auto                               image_size = decoded->get_image_size();
            shared_ptr<augment::image::params> params =
                factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
            max_cropbox_area  = params->cropbox.area();
            max_cropbox_ratio = max_cropbox_area / source_image_area;
        }
        {
            aug["scale"] = {0.3, 0.3};
            image::config                 cfg{js};
            image::extractor              extractor{cfg};
            augment::image::param_factory factory(aug);

            auto decoded = extractor.extract(image_data.data(), image_data.size());

            auto                               image_size = decoded->get_image_size();
            shared_ptr<augment::image::params> params =
                factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
            float cropbox_area  = params->cropbox.area();
            float cropbox_ratio = cropbox_area / source_image_area;
            EXPECT_NEAR(0.3, cropbox_ratio, 0.0001);
        }
        {
            aug["scale"] = {0.8, 0.8};
            image::config                 cfg{js};
            image::extractor              extractor{cfg};
            augment::image::param_factory factory(aug);

            auto decoded = extractor.extract(image_data.data(), image_data.size());

            auto                               image_size = decoded->get_image_size();
            shared_ptr<augment::image::params> params =
                factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
            float cropbox_area  = params->cropbox.area();
            float cropbox_ratio = cropbox_area / source_image_area;
            EXPECT_FLOAT_EQ(max_cropbox_ratio, cropbox_ratio);
        }
    }
}

TEST(image, decoded_image)
{
    cv::Mat img1 = cv::Mat(256, 256, CV_8UC3);

    image::decoded decoded(img1);
}

// TEST(image, image_config)
//{
//    nlohmann::json js = {{"min_size",300},{"max_size",400},{"channels",3},{"flip_enable", false}};

//    image::config config(js);
//    EXPECT_EQ(300,config.min_size);
//    EXPECT_EQ(400,config.max_size);
//    EXPECT_TRUE(config.channel_major);
//    EXPECT_EQ(3,config.channels);

//    EXPECT_FLOAT_EQ(0.0,config.flip_distribution.p());
//}

TEST(image, var_resize)
{
    auto                  mat = cv::Mat(200, 300, CV_8UC3);
    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {{"width", 400}, {"height", 400}, {"channels", 3}};
    nlohmann::json aug = {{"type", "image"}, {"fixed_aspect_ratio", true}, {"crop_enable", false}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    {
        cv::Mat image = decoded->get_image(0);
        EXPECT_EQ(300, image.size().width);
        EXPECT_EQ(200, image.size().height);
    }

    augment::image::param_factory      factory(aug);
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr = factory.make_params(
        image_size.width, image_size.height, config_ptr.width, config_ptr.height);

    image::transformer         trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(400, image.size().width);
    EXPECT_EQ(267, image.size().height);
}

TEST(image, var_resize_fixed_scale)
{
    auto                  mat = cv::Mat(200, 300, CV_8UC3);
    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {{"width", 400}, {"height", 400}, {"channels", 3}};
    nlohmann::json aug      = {{"type", "image"},
                          {"fixed_aspect_ratio", true},
                          {"crop_enable", false},
                          {"fixed_scaling_factor", 1.0}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    {
        cv::Mat image = decoded->get_image(0);
        EXPECT_EQ(300, image.size().width);
        EXPECT_EQ(200, image.size().height);
    }

    augment::image::param_factory      factory(aug);
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr = factory.make_params(
        image_size.width, image_size.height, config_ptr.width, config_ptr.height);

    image::transformer         trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(300, image.size().width);
    EXPECT_EQ(200, image.size().height);
}

TEST(image, warp_resize)
{
    int input_width   = 100;
    int input_height  = 200;
    int output_width  = 400;
    int output_height = 400;

    cv::Mat mat = generate_stripped_image(input_width, input_height, 0xFF, 0xFF00);

    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {{"width", input_width}, {"height", input_height}, {"channels", 3}};
    nlohmann::json aug      = {{"type", "image"}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    {
        cv::Mat image = decoded->get_image(0);
        EXPECT_EQ(input_width, image.size().width);
        EXPECT_EQ(input_height, image.size().height);
    }

    augment::image::param_factory      factory(aug);
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr =
        factory.make_ssd_params(image_size.width,
                                image_size.height,
                                output_width,
                                output_height,
                                vector<boundingbox::box>());

    image::transformer         trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(output_width, image.size().width);
    EXPECT_EQ(output_height, image.size().height);

    //cv::imwrite("image_resize_source.png", mat);
    //cv::imwrite("image_resize.png", image);

    const int blurred_bias = 5;
    uint8_t*  data         = image.data;
    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width / 2 - blurred_bias; j++)
        {
            EXPECT_EQ(data[0], 0xFF);
            EXPECT_EQ(data[1], 0x00);
            EXPECT_EQ(data[2], 0x00);
            data += 3;
        }
        for (int j = 0; j < blurred_bias * 2; j++)
        {
            data += 3;
        }
        for (int j = 0; j < output_width / 2 - blurred_bias; j++)
        {
            EXPECT_EQ(data[0], 0x00);
            EXPECT_EQ(data[1], 0xFF);
            EXPECT_EQ(data[2], 0x00);
            data += 3;
        }
    }
}

TEST(image, expand_not)
{
    auto                  mat = cv::Mat(300, 300, CV_8UC3);
    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {
        {"width", mat.size().width}, {"height", mat.size().height}, {"channels", 3}};
    nlohmann::json aug = {{"type", "image"},
                          {"expand_probability", 0.0},
                          {"expand_ratio", {5, 10}},
                          {"crop_enable", false}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    augment::image::param_factory      factory(aug);
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr = factory.make_params(
        image_size.width, image_size.height, config_ptr.width, config_ptr.height);

    EXPECT_FLOAT_EQ(params_ptr->expand_ratio, 1.0);
}

TEST(image, expand_ratio_invalid)
{
    auto                  mat = cv::Mat(300, 200, CV_8UC3);
    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {
        {"width", mat.size().width}, {"height", mat.size().height}, {"channels", 3}};
    nlohmann::json aug = {{"type", "image"},
                          {"expand_probability", 1.0},
                          {"expand_ratio", {0.01, 0.99}},
                          {"crop_enable", false}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());
    EXPECT_THROW(augment::image::param_factory{aug}, std::invalid_argument);
}

TEST(image, expand)
{
    cv::Scalar            color(0, 0, 255);
    auto                  mat = cv::Mat(300, 200, CV_8UC3, color);
    vector<unsigned char> img;
    cv::imencode(".png", mat, img);

    nlohmann::json jsConfig = {
        {"width", mat.size().width}, {"height", mat.size().height}, {"channels", 3}};
    nlohmann::json aug = {{"type", "image"},
                          {"expand_probability", 1.0},
                          {"expand_ratio", {4.00, 4.00}},
                          {"crop_enable", false}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    augment::image::param_factory      factory(aug);
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr =
        factory.make_ssd_params(image_size.width,
                                image_size.height,
                                config_ptr.width * 4,
                                config_ptr.height * 4,
                                vector<boundingbox::box>());

    image::transformer         trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat  image          = transformed->get_image(0);
    cv::Mat  expected_image = cv::Mat::zeros(image.size(), mat.type());
    cv::Rect roi(params_ptr->expand_offset, mat.size());
    expected_image(roi).setTo(color);
    auto the_sum = cv::sum(image != expected_image);
    EXPECT_EQ(the_sum, cv::Scalar(0, 0, 0, 0)) << "Expanded image differs from the original";
}

TEST(image, var_transform_flip)
{
    auto                  indexed = generate_indexed_image(256, 256);
    vector<unsigned char> img;
    cv::imencode(".png", indexed, img);
    nlohmann::json jsConfig = {{"width", 256}, {"height", 256}, {"channels", 3}};
    nlohmann::json aug = {{"type", "image"}, {"fixed_aspect_ratio", true}, {"crop_enable", false}};

    image::config config_ptr{jsConfig};

    image::extractor           ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    std::default_random_engine    dre;
    augment::image::param_factory factory(aug);

    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params_ptr = factory.make_params(
        image_size.width, image_size.height, config_ptr.width, config_ptr.height);
    params_ptr->flip = true;

    image::transformer         trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(256, image.size().width);
    EXPECT_EQ(256, image.size().height);

    EXPECT_TRUE(check_value(transformed, 0, 0, 255, 0));
    EXPECT_TRUE(check_value(transformed, 100, 100, 255 - 100, 100));
}

bool test_contrast_image(cv::Mat m, float v1, float v2, float v3)
{
    bool     rc = true;
    uint8_t* p  = m.data;
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v1;
            rc &= *p++ == v1;
            rc &= *p++ == v1;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v2;
            rc &= *p++ == v2;
            rc &= *p++ == v2;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v3;
            rc &= *p++ == v3;
            rc &= *p++ == v3;
        }
    }

    if (!rc)
    {
        INFO << m.at<cv::Vec3b>(0, 0);
        INFO << m.at<cv::Vec3b>(128, 0);
        INFO << m.at<cv::Vec3b>(256, 0);
    }

    return rc;
}

TEST(photometric, contrast)
{
    cv::Mat  source{384, 512, CV_8UC3};
    uint8_t* p = source.data;
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 0;
            *p++ = 0;
            *p++ = 0;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 127;
            *p++ = 127;
            *p++ = 127;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 255;
            *p++ = 255;
            *p++ = 255;
        }
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 1.0);
        EXPECT_TRUE(test_contrast_image(mat, 0, 127, 255));
    }

    {
        float   cscale = 0.5;
        cv::Mat mat    = source.clone();
        image::photometric::cbsjitter(mat, cscale, 1.0, 1.0);
        EXPECT_TRUE(test_contrast_image(mat, 64, 127, 191));
    }

    {
        float   cscale = 0.1;
        cv::Mat mat    = source.clone();
        image::photometric::cbsjitter(mat, cscale, 1.0, 1.0);
        EXPECT_TRUE(test_contrast_image(mat, 115, 127, 140));
    }
}

TEST(photometric, brightness)
{
    cv::Mat  source{384, 512, CV_8UC3};
    uint8_t* p = source.data;
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 0;
            *p++ = 0;
            *p++ = 0;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 127;
            *p++ = 127;
            *p++ = 127;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 255;
            *p++ = 255;
            *p++ = 255;
        }
    }

    cv::imwrite("brightness_source.png", source);
    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 1.0);
        cv::imwrite("brightness_1_0_.png", mat);
        EXPECT_TRUE(test_contrast_image(mat, 0, 127, 255));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 0.5, 1.0);
        cv::imwrite("brightness_0_5.png", mat);
        EXPECT_TRUE(test_contrast_image(mat, 0, 64, 128));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 0.1, 1.0);
        cv::imwrite("brightness_0_1.png", mat);
        EXPECT_TRUE(test_contrast_image(mat, 0, 13, 26));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.5, 1.0);
        cv::imwrite("brightness_1_5.png", mat);
        EXPECT_TRUE(test_contrast_image(mat, 0, 190, 255));
    }
}

bool test_saturation(cv::Mat m, vector<float> v1, vector<float> v2, vector<float> v3)
{
    bool     rc = true;
    uint8_t* p  = m.data;
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v1[0];
            rc &= *p++ == v1[1];
            rc &= *p++ == v1[2];
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v2[0];
            rc &= *p++ == v2[1];
            rc &= *p++ == v2[2];
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            rc &= *p++ == v3[0];
            rc &= *p++ == v3[1];
            rc &= *p++ == v3[2];
        }
    }

    if (!rc)
    {
        INFO << m.at<cv::Vec3b>(0, 0);
        INFO << m.at<cv::Vec3b>(128, 0);
        INFO << m.at<cv::Vec3b>(256, 0);
    }

    return rc;
}

TEST(DISABLED_photometric, saturation)
{
    cv::Mat  source{128 * 4, 512, CV_8UC3};
    uint8_t* p = source.data;
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 128;
            *p++ = 0;
            *p++ = 0;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 0;
            *p++ = 128;
            *p++ = 0;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 0;
            *p++ = 0;
            *p++ = 128;
        }
    }
    for (int row = 0; row < 128; row++)
    {
        for (int col = 0; col < 512; col++)
        {
            *p++ = 128;
            *p++ = 128;
            *p++ = 128;
        }
    }

    cv::imwrite("saturation_source.png", source);
    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 1.0);
        cv::imwrite("saturation_1_0.png", mat);
        EXPECT_TRUE(test_saturation(mat, {128, 0, 0}, {0, 128, 0}, {0, 0, 128}));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 0.5);
        cv::imwrite("saturation_0_5.png", mat);
        EXPECT_TRUE(test_saturation(mat, {128, 64, 64}, {64, 128, 64}, {64, 64, 128}));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 0.1);
        cv::imwrite("saturation_0_1.png", mat);
        EXPECT_TRUE(test_saturation(mat, {128, 115, 115}, {115, 128, 115}, {115, 115, 128}));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 0.5);
        image::photometric::cbsjitter(mat, 1.0, 1.0, 1.5);
        cv::imwrite("saturation_1_5.png", mat);
        EXPECT_TRUE(test_saturation(mat, {128, 32, 32}, {32, 128, 32}, {32, 32, 128}));
    }

    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 0.0);
        cv::imwrite("saturation_0_0.png", mat);
        EXPECT_TRUE(test_saturation(mat, {128, 128, 128}, {128, 128, 128}, {128, 128, 128}));
    }
}

cv::Mat refHueShift(const cv::Mat image, int hue)
{
    cv::Mat hsv;
    cv::Mat ret;
    cv::cvtColor(image, hsv, CV_BGR2HSV);
    if (hue != 0)
    {
        uint8_t* p = hsv.data;
        for (int i = 0; i < hsv.size().area(); i++)
        {
            *p = (*p + hue) % 180;
            p += 3;
        }
    }
    cv::cvtColor(hsv, ret, CV_HSV2BGR);
    return ret;
}

TEST(photometric, hue)
{
    cv::Mat  source{1, 3, CV_8UC3, cv::Scalar::all(255)};
    uint8_t* p = source.data;
    for (int row = 0; row < 1; row++)
    {
        for (int col = 0; col < 1; col++)
        {
            *p++ = 128;
            *p++ = 0;
            *p++ = 0;
        }
    }
    for (int row = 0; row < 1; row++)
    {
        for (int col = 1; col < 2; col++)
        {
            *p++ = 0;
            *p++ = 128;
            *p++ = 255;
        }
    }

    for (int i = 0; i <= 180; i += 45 / 2)
    {
        cv::Mat mat = source.clone();
        image::photometric::cbsjitter(mat, 1.0, 1.0, 1.0, i);
        cv::Mat expected = refHueShift(source, i);
        auto lambda      = [](cv::Mat m) {
            stringstream ss;
            ss << m;
            return ss.str();
        };
        EXPECT_EQ(lambda(mat), lambda(expected)) << "at hue shift: " << i;
        //        string name = "hue_" + to_string(i) + ".png";
        //        cv::imwrite(name, mat);
    }
}
