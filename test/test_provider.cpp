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

#include <numeric>

#include "gtest/gtest.h"

#include "cpio.hpp"
#include "etl_boundingbox.hpp"
#include "etl_image.hpp"
#include "etl_label_map.hpp"
#include "file_util.hpp"
#include "gen_image.hpp"
#include "helpers.hpp"
#include "json.hpp"
#include "loader.hpp"
#include "provider_factory.hpp"
#include "provider_factory.hpp"
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
        media->provide(i, bp, out_buf);

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

TEST(provider, blob)
{
    const int      width  = 480;
    const int      height = 360;
    nlohmann::json image1 = {{"type", "image"},
                             {"name", "left"},
                             {"channel_major", false},
                             {"height", height},
                             {"width", width}};
    nlohmann::json image2 = {{"type", "image"},
                             {"name", "right"},
                             {"channel_major", false},
                             {"height", height},
                             {"width", width}};
    nlohmann::json blob = {
        {"type", "blob"}, {"output_type", "float"}, {"output_count", width * height}};
    nlohmann::json js = {{"etl", {image1, image2, blob}}};

    vector<char> input_left  = file_util::read_file_contents(CURDIR "/test_data/img_2112_70.jpg");
    vector<char> input_right = file_util::read_file_contents(CURDIR "/test_data/img_2112_70.jpg");

    // flip input_left
    vector<uint8_t> tmp;
    auto            mat        = cv::imdecode(input_left, CV_LOAD_IMAGE_COLOR);
    cv::Size        image_size = mat.size();
    cv::Mat         flipped;
    cv::flip(mat, flipped, 1);
    cv::imencode(".jpg", flipped, tmp);
    input_left.clear();
    input_left.insert(input_left.begin(), tmp.begin(), tmp.end());

    // generate blob data, same size as image
    vector<float> target_data{width * height};
    target_data.resize(width * height);
    iota(target_data.begin(), target_data.end(), 0);
    vector<char> target_cdata;
    char*        p = (char*)target_data.data();
    for (int i = 0; i < target_data.size() * sizeof(float); i++)
    {
        target_cdata.push_back(*p++);
    }

    // setup input and output buffers
    auto media = nervana::provider_factory::create(js);
    ASSERT_NE(nullptr, media);
    auto oshapes = media->get_output_shapes();

    auto buf_names = media->get_buffer_names();
    ASSERT_EQ(3, buf_names.size());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "left.image"), buf_names.end());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "right.image"), buf_names.end());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "blob"), buf_names.end());

    size_t left_size  = media->get_output_shape("left.image").get_byte_size();
    size_t right_size = media->get_output_shape("right.image").get_byte_size();
    size_t blob_size  = media->get_output_shape("blob").get_byte_size();
    ASSERT_EQ(image_size.area() * 3, left_size);
    ASSERT_EQ(image_size.area() * 3, right_size);
    ASSERT_EQ(480 * 360 * sizeof(float), blob_size);

    size_t batch_size = 1;

    fixed_buffer_map out_buf(oshapes, batch_size);

    encoded_record_list in_buf;
    encoded_record      record;
    record.add_element(input_left);
    record.add_element(input_right);
    record.add_element(target_cdata);
    in_buf.add_record(record);

    // call the provider
    media->provide(0, in_buf, out_buf);

    cv::Mat output_left{image_size, CV_8UC3, out_buf["left.image"]->data()};
    cv::imwrite("output_left.jpg", output_left);
    EXPECT_EQ(image_size, output_left.size());

    cv::Mat output_right{image_size, CV_8UC3, out_buf["right.image"]->data()};
    cv::imwrite("output_right.jpg", output_right);
    EXPECT_EQ(image_size, output_right.size());

    char* fp = out_buf["blob"]->data();
    for (int i = 0; i < target_data.size(); i++)
    {
        ASSERT_FLOAT_EQ(target_data[i], unpack<float>(&fp[i * sizeof(float)]));
    }
}

TEST(provider, char_map)
{
    string alphabet      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()敏捷的棕色狐狸跳過了懶狗";
    string transcript1   = "The quick brown fox jumps over the lazy dog";
    string transcript2   = "敏捷的棕色狐狸跳過了懶狗";
    size_t max_length    = 100;
    size_t batch_size    = 128;
    int    unknown_value = to_wstring(alphabet).size();

    nlohmann::json char_map = {{"type", "char_map"},
                               {"alphabet", alphabet},
                               {"max_length", max_length},
                               {"unknown_value", unknown_value}};
    nlohmann::json label = {{"type", "label"}, {"binary", true}};
    nlohmann::json js    = {{"etl", {char_map, label}}};

    auto media   = nervana::provider_factory::create(js);
    auto oshapes = media->get_output_shapes();

    fixed_buffer_map out_buf(oshapes, batch_size);

    encoded_record_list in_buf;
    encoded_record      record1 = create_transcript_record(transcript1, 0);
    encoded_record      record2 = create_transcript_record(transcript2, 1);
    for (int i = 0; i < batch_size; i++)
    {
        in_buf.add_record(i % 2 ? record2 : record1);
    }

    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, in_buf, out_buf);
    }

    for (int i = 0; i < batch_size; i++)
    {
        int target_value = unpack<int>(out_buf["label"]->get_item(i));
        EXPECT_EQ(i % 2, target_value);
    }
}

namespace
{
    encoded_record create_transcript_record(const string& transcript, int label)
    {
        encoded_record record;
        size_t         max_size = transcript.size();
        record.add_element(static_cast<const void*>(transcript.c_str()), max_size);
        record.add_element(&label, 4);

        return record;
    }
}
