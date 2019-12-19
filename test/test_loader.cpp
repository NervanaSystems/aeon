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

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <iostream>
#include <ios>

#include "gtest/gtest.h"

#define private public

#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"
#include "file_util.hpp"
#include "util.hpp"

#if defined(ENABLE_AEON_SERVICE)
#include "service/service.hpp"
#include "client/loader_remote.hpp"
#endif

using namespace std;
using namespace nervana;

using nlohmann::json;

namespace
{
    string create_manifest_file(size_t record_count, size_t width, size_t height)
    {
        string           manifest_filename = file_util::tmp_filename();
        manifest_builder mb;
        auto& ms = mb.record_count(record_count).image_width(width).image_height(height).create();
        ofstream f(manifest_filename);
        f << ms.str();
        return manifest_filename;
    }

#if defined(ENABLE_AEON_SERVICE)

    json create_some_config_with_manifest()
    {
        int    height            = 32;
        int    width             = 32;
        size_t batch_size        = 2;
        size_t record_count      = 10;
        string manifest_filename = create_manifest_file(record_count, width, height);

        json image = {{"type", "image"},
                      {"name", "image1"},
                      {"height", height},
                      {"width", width},
                      {"channel_major", false}};
        json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
        json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
        json js           = {{"manifest_filename", manifest_filename},
                   {"batch_size", batch_size},
                   {"iteration_mode", "ONCE"},
                   {"etl", {image, label}},
                   {"augmentation", augmentation}};
        return js;
    }

#endif
} // namespace

TEST(loader, syntax)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 1;
    string manifest_root = string(CURDIR) + "/test_data";
    string manifest      = manifest_root + "/manifest.tsv";

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {
        {{"type", "image"}, {"height", height}, {"width", width}, {"flip_enable", true}}};
    json js = {{"manifest_root", manifest_root},
               {"manifest_filename", manifest},
               {"batch_size", batch_size},
               {"iteration_mode", "INFINITE"},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);
}

TEST(loader, iterator)
{
    int    height            = 32;
    int    width             = 32;
    size_t batch_size        = 1;
    size_t record_count      = 10;
    string manifest_filename = create_manifest_file(record_count, width, height);

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"iteration_mode", "ONCE"},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    auto begin = train_set->begin();
    auto end   = train_set->end();

    EXPECT_NE(begin, end);
    EXPECT_EQ(0, begin.position());
    begin++;
    EXPECT_NE(begin, end);
    EXPECT_EQ(1, begin.position());
    ++begin;
    EXPECT_NE(begin, end);
    EXPECT_EQ(2, begin.position());
    for (int i = 2; i < record_count; i++)
    {
        EXPECT_NE(begin, end);
        begin++;
    }
    EXPECT_EQ(record_count, begin.position());
    EXPECT_EQ(begin, end);
}

TEST(loader, once)
{
    int    height            = 32;
    int    width             = 32;
    size_t batch_size        = 2;
    size_t record_count      = 10;
    string manifest_filename = create_manifest_file(record_count, width, height);

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"iteration_mode", "ONCE"},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    int count = 0;
    for (const fixed_buffer_map& data : *train_set)
    {
        (void)data; // silence compiler warning
        ASSERT_NE(count, record_count);
        count++;
    }
    ASSERT_EQ(record_count / 2, count);
}

TEST(loader, count)
{
    int    height            = 32;
    int    width             = 32;
    size_t batch_size        = 1;
    size_t record_count      = 10;
    string manifest_filename = create_manifest_file(record_count, width, height);

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"iteration_mode", "COUNT"},
               {"iteration_mode_count", 4},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    int expected_iterations = 4;

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    int count = 0;
    ASSERT_EQ(expected_iterations, train_set->batch_count());
    for (const fixed_buffer_map& data : *train_set)
    {
        (void)data; // silence compiler warning
        ASSERT_NE(count, record_count);
        count++;
    }
    ASSERT_EQ(expected_iterations, count);
}

TEST(loader, infinite)
{
    int    height            = 32;
    int    width             = 32;
    size_t batch_size        = 1;
    size_t record_count      = 10;
    string manifest_filename = create_manifest_file(record_count, width, height);

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"iteration_mode", "INFINITE"},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    int count               = 0;
    int expected_iterations = record_count * 3;
    for (const fixed_buffer_map& data : *train_set)
    {
        (void)data; // silence compiler warning
        count++;
        if (count == expected_iterations)
        {
            break;
        }
    }
    ASSERT_EQ(expected_iterations, count);
}

TEST(loader, cache)
{
    int    height            = 16;
    int    width             = 16;
    size_t batch_size        = 32;
    size_t record_count      = 1002;
    size_t block_size        = 300;
    string cache_root        = file_util::get_temp_directory();
    string manifest_filename = create_manifest_file(record_count, width, height);

    json image = {{"type", "image"},
                  {"name", "image1"},
                  {"height", height},
                  {"width", width},
                  {"channel_major", false}};
    json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"block_size", block_size},
               {"cache_directory", cache_root},
               {"iteration_mode", "INFINITE"},
               {"etl", {image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    int count               = 0;
    int expected_iterations = record_count * 3;
    for (const fixed_buffer_map& data : *train_set)
    {
        (void)data; // silence compiler warning
        count++;
        if (count == expected_iterations)
        {
            break;
        }
    }
    ASSERT_EQ(expected_iterations, count);
}

TEST(loader, test)
{
    int    height            = 16;
    int    width             = 16;
    size_t batch_size        = 32;
    size_t record_count      = 1003;
    size_t block_size        = 300;
    string manifest_filename = create_manifest_file(record_count, width, height);

    json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label        = {{"type", "label"}, {"binary", false}};
    json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    json js           = {{"manifest_filename", manifest_filename},
               {"batch_size", batch_size},
               {"block_size", block_size},
               {"etl", {js_image, label}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    auto buf_names = train_set->get_buffer_names();
    EXPECT_EQ(2, buf_names.size());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "image"), buf_names.end());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

    auto image_shape = train_set->get_shape("image");
    auto label_shape = train_set->get_shape("label");

    ASSERT_EQ(3, image_shape.size());
    EXPECT_EQ(height, image_shape[0]);
    EXPECT_EQ(width, image_shape[1]);
    EXPECT_EQ(3, image_shape[2]);

    ASSERT_EQ(1, label_shape.size());
    EXPECT_EQ(1, label_shape[0]);

    int count       = 0;
    int expected_id = 0;
    for (const fixed_buffer_map& data : *train_set)
    {
        ASSERT_EQ(2, data.size());
        const buffer_fixed_size_elements* image_buffer_ptr = data["image"];
        ASSERT_NE(nullptr, image_buffer_ptr);
        const buffer_fixed_size_elements& image_buffer = *image_buffer_ptr;
        for (int i = 0; i < batch_size; i++)
        {
            const char* image_data = image_buffer.get_item(i);
            cv::Mat     image{height, width, CV_8UC3, (char*)image_data};
            int         actual_id = embedded_id_image::read_embedded_id(image);
            // INFO << "train_loop " << expected_id << "," << actual_id;
            ASSERT_EQ(expected_id % record_count, actual_id);
            expected_id++;
        }

        if (count++ == 8)
        {
            break;
        }
    }
}

TEST(loader, provider)
{
    int    height       = 16;
    int    width        = 16;
    size_t batch_size   = 32;
    size_t record_count = 1003;
    string manifest     = create_manifest_file(record_count, width, height);

    json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label_config = {{"type", "label"}, {"binary", false}};
    json augmentation = {
        {{"height", height}, {"width", width}, {"type", "image"}, {"flip_enable", true}}};
    json js = {{"cpu_list", "0"},
               {"manifest_filename", manifest},
               {"batch_size", batch_size},
               {"iteration_mode", "INFINITE"},
               {"etl", {image_config, label_config}},
               {"augmentation", augmentation}};

    loader_factory factory;
    auto           train_set = factory.get_loader(js);

    auto buf_names = train_set->get_buffer_names();
    EXPECT_EQ(2, buf_names.size());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "image"), buf_names.end());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

    auto image_shape = train_set->get_shape("image");
    auto label_shape = train_set->get_shape("label");

    ASSERT_EQ(3, image_shape.size());
    EXPECT_EQ(height, image_shape[0]);
    EXPECT_EQ(width, image_shape[1]);
    EXPECT_EQ(3, image_shape[2]);

    ASSERT_EQ(1, label_shape.size());
    EXPECT_EQ(1, label_shape[0]);

    int count       = 0;
    int expected_id = 0;
    for (const fixed_buffer_map& data : *train_set)
    {
        ASSERT_EQ(2, data.size());
        const buffer_fixed_size_elements* image_buffer_ptr = data["image"];
        ASSERT_NE(nullptr, image_buffer_ptr);
        const buffer_fixed_size_elements& image_buffer = *image_buffer_ptr;
        for (int i = 0; i < batch_size; i++)
        {
            const char* image_data = image_buffer.get_item(i);
            cv::Mat     image{height, width, CV_8UC3, (char*)image_data};
            int         actual_id = embedded_id_image::read_embedded_id(image);
            // INFO << "train_loop " << expected_id << "," << actual_id;
            ASSERT_EQ(expected_id % record_count, actual_id);
            expected_id++;
        }

        if (count++ == 8)
        {
            break;
        }
    }
}

static std::string generate_manifest_file(
    size_t record_count,
    std::vector<std::string> image_files = std::vector<std::string>{})
{
    std::string   manifest_name = "manifest.txt";
    if (image_files.empty())
    {
        image_files = std::vector<std::string>{"flowers.jpg", "img_2112_70.jpg"};
    }
    const std::size_t img_count = image_files.size();
    std::ofstream f(manifest_name);
    if (f)
    {
        f << nervana::manifest_file::get_metadata_char();
        f << nervana::manifest_file::get_file_type_id();
        f << nervana::manifest_file::get_delimiter();
        f << nervana::manifest_file::get_string_type_id();
        f << "\n";
        for (size_t i = 0; i < record_count; i++)
        {
            f << image_files[i % img_count];
            f << nervana::manifest_file::get_delimiter();
            f << std::to_string(i % img_count);
            f << "\n";
        }
    }
    return manifest_name;
}

TEST(DISABLED_loader, deterministic)
{
    std::string test_data_directory = file_util::path_join(string(CURDIR), "test_data");
    std::string manifest            = generate_manifest_file(20);

    int    height     = 1;
    int    width      = 1;
    size_t batch_size = 4;

    const uint32_t seed = 1234;

    nlohmann::json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};

    nlohmann::json label_config = {{"type", "label"}, {"binary", false}};
    auto           aug_config   = vector<nlohmann::json>{{{"type", "image"},
                                              {"scale", {0.5, 1.0}},
                                              {"saturation", {0.5, 2.0}},
                                              {"contrast", {0.5, 1.0}},
                                              {"brightness", {0.5, 1.0}},
                                              {"flip_enable", true},
                                            }};
    nlohmann::json config = {{"manifest_root", test_data_directory},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"cpu_list", ""},
                             {"shuffle_manifest", true},
                             {"etl", {image_config, label_config}},
                             {"augmentation", aug_config},
                             {"random_seed", seed}};

    auto loader = nervana::loader_local{config};
    loader.get_current_iter();
    auto& buffer = *loader.get_current_iter();

    const uint32_t expected_result[3] = {1210130715, 639721302, 1414547238};
    uint32_t*      data               = reinterpret_cast<uint32_t*>(buffer["image"]->data());
    EXPECT_EQ(data[0], expected_result[0]);
    EXPECT_EQ(data[1], expected_result[1]);
    EXPECT_EQ(data[2], expected_result[2]);
}

#if defined(ENABLE_AEON_SERVICE)
TEST(loader, loader_factory_no_remote)
{
    loader_factory factory;
    json           config_json = create_some_config_with_manifest();
    string         config      = config_json.dump();

    unique_ptr<loader> ptr = factory.get_loader(config);
    ASSERT_TRUE(dynamic_cast<loader_local*>(ptr.get()) != 0);

    unique_ptr<loader> ptr2 = factory.get_loader(config);
    ASSERT_TRUE(dynamic_cast<loader_remote*>(ptr2.get()) == 0);
}

TEST(loader, loader_factory_remote)
{
    loader_factory factory;
    json           config_json = create_some_config_with_manifest();
    aeon::service  service{"http://127.0.0.1:34568"};

    // there is no service running, so we expect exception
    config_json["remote"] = {{"address", "127.0.0.1"}, {"port", 34569}};
    EXPECT_THROW(unique_ptr<loader> ptr = factory.get_loader(config_json), std::runtime_error);

    config_json["remote"]   = {{"address", "127.0.0.1"}, {"port", 34568}};
    unique_ptr<loader> ptr2 = factory.get_loader(config_json);
    ASSERT_TRUE(dynamic_cast<loader_remote*>(ptr2.get()) != 0);
    ASSERT_TRUE(dynamic_cast<loader_local*>(ptr2.get()) == 0);
}
#endif
