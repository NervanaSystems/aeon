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

using namespace std;
using namespace nervana;

static string create_manifest_file(size_t record_count, size_t width, size_t height)
{
    string           manifest_filename = file_util::tmp_filename();
    manifest_builder mb;
    auto&    ms = mb.record_count(record_count).image_width(width).image_height(height).create();
    ofstream f(manifest_filename);
    f << ms.str();
    return manifest_filename;
}

TEST(loader, syntax)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 1;
    string manifest_root = string(CURDIR) + "/test_data";
    string manifest      = manifest_root + "/manifest.csv";

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {
        {{"type", "image"}, {"height", height}, {"width", width}, {"flip_enable", true}}};
    nlohmann::json js = {{"manifest_root", manifest_root},
                         {"manifest_filename", manifest},
                         {"batch_size", batch_size},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    auto train_set = loader{js};
}

TEST(loader, iterator)
{
    int    height            = 32;
    int    width             = 32;
    size_t batch_size        = 1;
    size_t record_count      = 10;
    string manifest_filename = create_manifest_file(record_count, width, height);

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "ONCE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    auto begin = train_set.begin();
    auto end   = train_set.end();

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

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "ONCE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    int count = 0;
    for (const fixed_buffer_map& data : train_set)
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

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "COUNT"},
                         {"iteration_mode_count", 4},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    int expected_iterations = 4;

    loader train_set{js};

    int count = 0;
    ASSERT_EQ(expected_iterations, train_set.m_batch_count_value);
    for (const fixed_buffer_map& data : train_set)
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

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    int count               = 0;
    int expected_iterations = record_count * 3;
    for (const fixed_buffer_map& data : train_set)
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

    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"block_size", block_size},
                         {"cache_directory", cache_root},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    int count               = 0;
    int expected_iterations = record_count * 3;
    for (const fixed_buffer_map& data : train_set)
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

    nlohmann::json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    nlohmann::json label        = {{"type", "label"}, {"binary", false}};
    nlohmann::json augmentation = {{{"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js           = {{"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"block_size", block_size},
                         {"etl", {js_image, label}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    auto buf_names = train_set.get_buffer_names();
    EXPECT_EQ(2, buf_names.size());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "image"), buf_names.end());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

    auto image_shape = train_set.get_shape("image");
    auto label_shape = train_set.get_shape("label");

    ASSERT_EQ(3, image_shape.size());
    EXPECT_EQ(height, image_shape[0]);
    EXPECT_EQ(width, image_shape[1]);
    EXPECT_EQ(3, image_shape[2]);

    ASSERT_EQ(1, label_shape.size());
    EXPECT_EQ(1, label_shape[0]);

    int count       = 0;
    int expected_id = 0;
    for (const fixed_buffer_map& data : train_set)
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

    nlohmann::json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    nlohmann::json label_config = {{"type", "label"}, {"binary", false}};
    nlohmann::json augmentation = {
        {{"height", height}, {"width", width}, {"type", "image"}, {"flip_enable", true}}};
    nlohmann::json js = {{"decode_thread_count", 1},
                         {"manifest_filename", manifest},
                         {"batch_size", batch_size},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image_config, label_config}},
                         {"augmentation", augmentation}};

    loader train_set{js};

    auto buf_names = train_set.get_buffer_names();
    EXPECT_EQ(2, buf_names.size());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "image"), buf_names.end());
    EXPECT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

    auto image_shape = train_set.get_shape("image");
    auto label_shape = train_set.get_shape("label");

    ASSERT_EQ(3, image_shape.size());
    EXPECT_EQ(height, image_shape[0]);
    EXPECT_EQ(width, image_shape[1]);
    EXPECT_EQ(3, image_shape[2]);

    ASSERT_EQ(1, label_shape.size());
    EXPECT_EQ(1, label_shape[0]);

    int count       = 0;
    int expected_id = 0;
    for (const fixed_buffer_map& data : train_set)
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

static std::string generate_manifest_file(size_t record_count)
{
    std::string   manifest_name = "manifest.txt";
    const char*   image_files[] = {"flowers.jpg", "img_2112_70.jpg"};
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
            f << image_files[i % 2];
            f << nervana::manifest_file::get_delimiter();
            f << std::to_string(i % 2);
            f << "\n";
        }
    }
    return manifest_name;
}

TEST(loader, deterministic)
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
                                              {"flip_enable", true}}};
    nlohmann::json config = {{"manifest_root", test_data_directory},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"decode_thread_count", 0},
                             {"shuffle_manifest", true},
                             {"etl", {image_config, label_config}},
                             {"augmentation", aug_config},
                             {"random_seed", seed}};

    auto loader = nervana::loader{config};
    loader.get_current_iter();
    auto& buffer = *loader.get_current_iter();

    const uint32_t expected_result[3] = {0x3f393834, 0x2c284d47, 0x5c59502b};
    uint32_t*      data               = reinterpret_cast<uint32_t*>(buffer["image"]->data());
    EXPECT_EQ(data[0], expected_result[0]);
    EXPECT_EQ(data[1], expected_result[1]);
    EXPECT_EQ(data[2], expected_result[2]);
}

TEST(benchmark, imagenet)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    char* cache_root    = getenv("TEST_IMAGENET_CACHE");

    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
    }
    else
    {
        int    height     = 224;
        int    width      = 224;
        size_t batch_size = 128;
        string manifest   = file_util::path_join(manifest_root, "train-index.tsv");

        nlohmann::json image_config = {
            {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
        nlohmann::json label_config = {{"type", "label"}, {"binary", false}};
        auto           aug_config   = vector<nlohmann::json>{{{"type", "image"},
                                                  {"scale", {0.5, 1.0}},
                                                  {"saturation", {0.5, 2.0}},
                                                  {"contrast", {0.5, 1.0}},
                                                  {"brightness", {0.5, 1.0}},
                                                  {"flip_enable", true}}};
        nlohmann::json config = {{"manifest_root", manifest_root},
                                 {"manifest_filename", manifest},
                                 {"batch_size", batch_size},
                                 {"iteration_mode", "INFINITE"},
                                 {"cache_directory", cache_root},
                                 {"decode_thread_count", 0},
                                 {"web_server_port", 8086},
                                 {"etl", {image_config, label_config}},
                                 {"augmentation", aug_config}};

        chrono::high_resolution_clock                     timer;
        chrono::time_point<chrono::high_resolution_clock> start_time;
        chrono::time_point<chrono::high_resolution_clock> zero_time;
        chrono::milliseconds                              total_time{0};
        try
        {
            auto train_set = nervana::loader{config};

            size_t       total_batch   = ceil((float)train_set.record_count() / (float)batch_size);
            size_t       current_batch = 0;
            const size_t batches_per_output = 10;
            for (const nervana::fixed_buffer_map& x : train_set)
            {
                (void)x;
                if (++current_batch % batches_per_output == 0)
                {
                    auto last_time = start_time;
                    start_time     = timer.now();
                    float ms_time =
                        chrono::duration_cast<chrono::milliseconds>(start_time - last_time).count();
                    float sec_time = ms_time / 1000.;
                    cout << "batch " << current_batch << " of " << total_batch;
                    if (last_time != zero_time)
                    {
                        cout << " time " << ms_time;
                        cout << " " << (float)batches_per_output / sec_time << " batches/s";
                        total_time +=
                            chrono::duration_cast<chrono::milliseconds>(start_time - last_time);
                        cout << "\t\taverage "
                             << (float)(current_batch - batches_per_output) /
                                    (total_time.count() / 1000.0f)
                             << " batches/s";
                    }
                    cout << endl;
                }

                // if (current_batch == 30) break;
            }
        }
        catch (exception& err)
        {
            cout << "error processing dataset" << endl;
            cout << err.what() << endl;
        }
    }
}

TEST(benchmark, decode_jpeg)
{
    stopwatch timer;
    size_t    manifest_size = 10000;
    string    image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");

    vector<char> image_data = file_util::read_file_contents(image_path);
    timer.start();
    for (size_t i = 0; i < manifest_size; i++)
    {
        cv::Mat output_img;
        cv::Mat input_img(1, image_data.size(), CV_8UC3, image_data.data());
        cv::imdecode(input_img, CV_8UC3, &output_img);
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << ((float)manifest_size / time) << " images/second " << endl;
}

TEST(benchmark, read_jpeg)
{
    stopwatch timer;

    size_t       manifest_size = 10000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    manifest_file manifest{manifest_stream, false};

    vector<vector<string>>* block = nullptr;
    size_t                  count = 0;
    size_t                  mod   = 10000;
    timer.start();
    for (block = manifest.next(); block != nullptr; block = manifest.next())
    {
        for (const vector<string>& record : *block)
        {
            count++;
            vector<char> image_data = file_util::read_file_contents(record[0]);
            cv::Mat      output_img;
            cv::Mat      input_img(1, image_data.size(), CV_8UC3, image_data.data());
            cv::imdecode(input_img, CV_8UC3, &output_img);
            if (count % mod == 0)
            {
                auto time = (float)timer.get_milliseconds() / 1000.;
                cout << ((float)count / time) << " images/second " << endl;
            }
        }
    }
}

TEST(benchmark, load_jpeg)
{
    stopwatch timer;

    size_t manifest_size = 10000;
    string image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");

    timer.start();
    for (size_t i = 0; i < manifest_size; i++)
    {
        auto data = file_util::read_file_contents(image_path);
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << "images/second " << ((float)manifest_size / time) << endl;
}

TEST(benchmark, load_jpeg_manifest)
{
    stopwatch timer;

    size_t       manifest_size = 10000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    manifest_file manifest{manifest_stream, false};

    vector<vector<string>>* block = nullptr;
    timer.start();
    for (block = manifest.next(); block != nullptr; block = manifest.next())
    {
        for (const vector<string>& record : *block)
        {
            auto data = file_util::read_file_contents(record[0]);
        }
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << "images/second " << ((float)manifest_size / time) << endl;
}

TEST(benchmark, load_block_manager)
{
    stopwatch timer;
    string    cache_directory = "/scratch/bob/aeon_cache";
    bool      shuffle         = false;
    size_t    block_size      = 5000;

    size_t       manifest_size = 30000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    manifest_file manifest{manifest_stream, false};

    block_loader_file loader{&manifest, block_size};

    block_manager manager{&loader, 5000, cache_directory, shuffle};

    encoded_record_list* records;
    timer.start();
    float  count      = 0;
    size_t iterations = (manifest_size / block_size) * 3;
    for (size_t i = 0; i < iterations; i++)
    {
        records = manager.next();
        timer.stop();
        count      = records->size();
        float time = timer.get_microseconds() / 1000000.;
        cout << "count=" << count << ", time=" << time << " images/second " << count / time << endl;
        timer.start();
    }
}
