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

#include <cmath>
#include <fstream>
#include <numeric>

#include "gtest/gtest.h"
#include <opencv2/imgproc/imgproc.hpp>

#include "cpio.hpp"
#include "etl_image.hpp"
#include "file_util.hpp"
#include "gen_image.hpp"
#include "helpers.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "provider_factory.hpp"
#include "provider.hpp"
#include "util.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

namespace
{
    encoded_record create_transcript_record(const string& transcript, int label);
}

TEST(provider, empty_config)
{
    nlohmann::json image = {{"type", "image"}, {"height", 1}, {"width", 1}};
    nlohmann::json label = {{"type", "label"}};
    nlohmann::json js    = {{"etl", {image, label}}};

    nervana::provider_factory::create(js);
}

TEST(provider, image)
{
    nlohmann::json image = {{"type", "image"}, {"height", 128}, {"width", 128}};
    nlohmann::json label = {{"type", "label"}, {"binary", true}};
    nlohmann::json js    = {{"etl", {image, label}}};

    auto media   = nervana::provider_factory::create(js);
    auto oshapes = media->get_output_shapes();

    size_t batch_size = 128;

    fixed_buffer_map    out_buf(oshapes, batch_size);
    encoded_record_list bp;

    auto files = image_dataset.get_files();
    ASSERT_NE(0, files.size());
    ifstream f(files[0], istream::binary);
    ASSERT_TRUE(f);
    cpio::reader reader(f);
    for (int i = 0; i < reader.record_count() / 2; i++)
    {
        reader.read(bp, 2);
    }

    EXPECT_GT(bp.size(), batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, bp.record(i), out_buf);

        //  cv::Mat mat(width,height,CV_8UC3,&dbuffer[0]);
        //  string filename = "data" + to_string(i) + ".png";
        //  cv::imwrite(filename,mat);
    }
    for (int i = 0; i < batch_size; i++)
    {
        int target_value = unpack<int>(out_buf["label"]->get_item(i));
        EXPECT_EQ(42 + i, target_value);
    }
}

TEST(provider, image_paddle_imagenet_training_augmentation)
{
    string test_data_directory = file_util::path_join(string(CURDIR), "test_data");
    string test_results_directory = file_util::path_join(string(CURDIR), "test_results");

    const size_t in_img_height = 360;
    const size_t in_img_width  = 480;

    const size_t height         = 224;
    const size_t width          = 224;
    const size_t crop_height    = 201;
    const size_t crop_width     = 171;
    const size_t channels       = 3;
    const size_t batch_size     = 1;
    const size_t elements_count = height * width * channels;

    nlohmann::json image_config = {{"height", height},
                                   {"width", width},
                                   {"channels", 3},
                                   {"output_type", "float"},
                                   {"channel_major", true},
                                   {"bgr_to_rgb", true},
                               };

    nlohmann::json aug_config = {{"type", "image"},
                                 {"flip_enable", true},
                                 {"center", false},
                                 {"crop_enable", true},
                                 {"interpolation_method", "LINEAR"},
                                 {"mean", {0.485, 0.456, 0.406}},
                                 {"stddev", {0.229, 0.224, 0.225}},
                                 {"resize_short_size", 0},
                                 {"debug_output_directory", test_results_directory},
                             };

    // --- prepare image augmentation parameters ---
    augment::image::param_factory factory(aug_config);
    image_params_builder builder(factory.make_params(in_img_width, in_img_height, width, height));
    shared_ptr<augment::image::params> aug_params_ptr =
            builder.cropbox(50, 50, crop_width, crop_height)
                   .flip(true)
                   .angle(0);

    augmentation data_augmentation;
    data_augmentation.m_image_augmentations = aug_params_ptr;

    // --- prepare output buffer ---
    shape_type out_shape{{channels, height, width}, output_type{"float"}};
    fixed_buffer_map out_buf;
    out_buf.add_item("image", out_shape, batch_size);

    // --- prepare input data ---
    string input_file_path = file_util::path_join(test_data_directory,
                                                  "img_2112_70.jpg");
    std::vector<char> input_data{file_util::read_binary_file<char>(input_file_path)};

    // --- extract, transform, load ---
    provider::image img_provider{image_config, aug_config};
    img_provider.provide(0, input_data, out_buf, data_augmentation);

    // ---compare results ---
    using pixel_type = float;
    pixel_type* output_image = reinterpret_cast<pixel_type*>(out_buf["image"]->data());
    const std::vector<pixel_type> expected_result =
        file_util::read_binary_file<pixel_type>(file_util::path_join(test_data_directory,
                                           "augment_output_linear_train.bin"));

    double err{0}, rel_err{0}, max_err{0}, max_rel_err{0};
    for (std::size_t i = 0; i < elements_count; ++i)
    {
        double diff = std::abs(expected_result[i] - output_image[i]);
        err += diff;
        rel_err += diff / std::abs(expected_result[i]);
        max_err = std::max(max_err, diff);
        max_rel_err = std::max(max_rel_err, diff / std::abs(expected_result[i]));
    }
    double avg_err = err / static_cast<double>(elements_count);
    double avg_rel_err = rel_err / static_cast<double>(elements_count);

    EXPECT_LE(avg_err, 1e-3f);
    EXPECT_LE(avg_rel_err, 1e-3f);
    EXPECT_LE(max_err, 1e-3f);
    EXPECT_LE(max_rel_err, 1e-2f);
}

TEST(provider, image_paddle_imagenet_validate_augmentation)
{
    string test_data_directory = file_util::path_join(string(CURDIR), "test_data");
    string test_results_directory = file_util::path_join(string(CURDIR), "test_results");

    const size_t in_img_height = 360;
    const size_t in_img_width  = 480;

    const size_t height         = 224;
    const size_t width          = 224;
    const size_t channels       = 3;
    const size_t batch_size     = 1;
    const size_t elements_count = height * width * channels;

    nlohmann::json image_config = {{"height", height},
                                   {"width", width},
                                   {"channels", 3},
                                   {"output_type", "float"},
                                   {"channel_major", true},
                                   {"bgr_to_rgb", true},
                               };

    const int resize_short_size = 256;
    const double scale = static_cast<double>(width) / resize_short_size;
    nlohmann::json aug_config = {{"type", "image"},
                                 {"flip_enable", false},
                                 {"center", true},
                                 {"crop_enable", true},
                                 {"scale", {scale, scale}},
                                 {"interpolation_method", "LINEAR"},
                                 {"mean", {0.485, 0.456, 0.406}},
                                 {"stddev", {0.229, 0.224, 0.225}},
                                 {"resize_short_size", resize_short_size},
                                 {"debug_output_directory", test_results_directory},
                             };

    // --- prepare image augmentation parameters ---
    augment::image::param_factory factory(aug_config);
    shared_ptr<augment::image::params> aug_params_ptr =
        factory.make_params(in_img_width, in_img_height, width, height);

    augmentation data_augmentation;
    data_augmentation.m_image_augmentations = aug_params_ptr;

    // --- prepare output buffer ---
    shape_type out_shape{{channels, height, width}, output_type{"float"}};
    fixed_buffer_map out_buf;
    out_buf.add_item("image", out_shape, batch_size);

    // --- prepare input data ---
    string input_file_path = file_util::path_join(test_data_directory,
                                                  "img_2112_70.jpg");
    std::vector<char> input_data{file_util::read_binary_file<char>(input_file_path)};

    // --- extract, transform, load ---
    provider::image img_provider{image_config, aug_config};
    img_provider.provide(0, input_data, out_buf, data_augmentation);

    // ---compare results ---
    using pixel_type = float;
    pixel_type* output_image = reinterpret_cast<pixel_type*>(out_buf["image"]->data());
    const std::vector<pixel_type> expected_result =
        file_util::read_binary_file<pixel_type>(file_util::path_join(test_data_directory,
                                           "augment_output_linear_eval.bin"));
    double err{0}, rel_err{0}, max_err{0}, max_rel_err{0};
    for (std::size_t i = 0; i < elements_count; ++i)
    {
        double diff = std::abs(expected_result[i] - output_image[i]);
        err += diff;
        rel_err += diff / std::abs(expected_result[i]);
        max_err = std::max(max_err, diff);
        max_rel_err = std::max(max_rel_err, diff / std::abs(expected_result[i]));
    }
    double avg_err = err / static_cast<double>(elements_count);
    double avg_rel_err = rel_err / static_cast<double>(elements_count);

    EXPECT_LE(avg_err, 1e-3f);
    EXPECT_LE(avg_rel_err, 1e-3f);
    EXPECT_LE(max_err, 1e-5f);
    EXPECT_LE(max_rel_err, 1e-2f);
}

TEST(provider, argtype)
{
    {
        /* Create extractor with default num channels param */
        string        cfgString = "{\"height\":10, \"width\":10}";
        auto          js        = nlohmann::json::parse(cfgString);
        image::config cfg{js};
        auto          ic = make_shared<image::extractor>(cfg);
        EXPECT_EQ(ic->get_channel_count(), 3);
    }

    {
        string cfgString = R"(
            {
                "height" : 30,
                "width" : 30
            }
        )";

        nlohmann::json js = nlohmann::json::parse(cfgString);
        nlohmann::json aug;
        image::config  itpj(js);

        // output the fixed parameters
        EXPECT_EQ(30, itpj.height);
        EXPECT_EQ(30, itpj.width);

        // output the random parameters
        default_random_engine         r_eng(0);
        augment::image::param_factory img_prm_maker(aug);
        auto                          imgt = make_shared<image::transformer>(itpj);

        auto input_img_ptr = make_shared<image::decoded>(cv::Mat(256, 320, CV_8UC3));

        auto image_size = input_img_ptr->get_image_size();
        auto its =
            img_prm_maker.make_params(image_size.width, image_size.height, itpj.width, itpj.height);
    }
}
