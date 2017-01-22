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

#include "gtest/gtest.h"

#define private public

#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"

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

TEST(loader, iteration_mode)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 1;
    string manifest_root = string(CURDIR) + "/test_data";
    string manifest      = manifest_root + "/manifest.csv";

    {
        nlohmann::json js = {{"type", "image,label"},
                             {"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"image",
                              {{"height", height},
                               {"width", width},
                               {"channel_major", false},
                               {"flip_enable", true}}},
                             {"label", {{"binary", false}}}};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type", "image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", manifest_root},
                             {"batch_size", batch_size},
                             {"iteration_mode", "ONCE"},
                             {"image",
                              {{"height", height},
                               {"width", width},
                               {"channel_major", false},
                               {"flip_enable", true}}},
                             {"label", {{"binary", false}}}};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type", "image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", manifest_root},
                             {"batch_size", batch_size},
                             {"iteration_mode", "COUNT"},
                             {"iteration_mode_count", 1000},
                             {"image",
                              {{"height", height},
                               {"width", width},
                               {"channel_major", false},
                               {"flip_enable", true}}},
                             {"label", {{"binary", false}}}};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type", "image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", manifest_root},
                             {"batch_size", batch_size},
                             {"iteration_mode", "COUNT"},
                             {"image",
                              {{"height", height},
                               {"width", width},
                               {"channel_major", false},
                               {"flip_enable", true}}},
                             {"label", {{"binary", false}}}};

        EXPECT_THROW(loader{js}, std::invalid_argument);
    }

    {
        nlohmann::json js = {{"type", "image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", manifest_root},
                             {"batch_size", batch_size},
                             {"iteration_mode", "BLAH"},
                             {"image",
                              {{"height", height},
                               {"width", width},
                               {"channel_major", false},
                               {"flip_enable", true}}},
                             {"label", {{"binary", false}}}};

        EXPECT_THROW(loader{js}, std::invalid_argument);
    }
}

TEST(loader, iterator)
{
    int            height            = 32;
    int            width             = 32;
    size_t         batch_size        = 1;
    size_t         record_count      = 10;
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

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
    int            height            = 32;
    int            width             = 32;
    size_t         batch_size        = 2;
    size_t         record_count      = 10;
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"iteration_mode", "ONCE"},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

    loader train_set{js};

    int count = 0;
    for (const fixed_buffer_map& data : train_set)
    {
        (void)data; // silence compiler warning
        ASSERT_NE(count, record_count);
        count++;
    }
    ASSERT_EQ(record_count/2, count);
}

TEST(loader, count)
{
    int            height            = 32;
    int            width             = 32;
    size_t         batch_size        = 1;
    size_t         record_count      = 10;
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"iteration_mode", "COUNT"},
        {"iteration_mode_count", 4},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

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
    int            height            = 32;
    int            width             = 32;
    size_t         batch_size        = 1;
    size_t         record_count      = 10;
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"iteration_mode", "INFINITE"},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

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
    int            height            = 16;
    int            width             = 16;
    size_t         batch_size        = 32;
    size_t         record_count      = 1002;
    size_t         block_size        = 300;
    string         cache_root        = file_util::get_temp_directory();
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"block_size", block_size},
        {"cache_directory", cache_root},
        {"iteration_mode", "INFINITE"},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

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
    int            height            = 16;
    int            width             = 16;
    size_t         batch_size        = 32;
    size_t         record_count      = 1003;
    size_t         block_size        = 300;
    string         manifest_filename = create_manifest_file(record_count, width, height);
    nlohmann::json js                = {
        {"type", "image,label"},
        {"manifest_filename", manifest_filename},
        {"batch_size", batch_size},
        {"block_size", block_size},
        {"image",
         {{"height", height}, {"width", width}, {"channel_major", false}, {"flip_enable", true}}},
        {"label", {{"binary", false}}}};

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
    for (const fixed_buffer_map& data :
         train_set) // if d1 created with infinite, this will just keep going
    {
        ASSERT_EQ(2, data.size());
        //        model_fit_one_iter(data);
        //        data['image'], data['label'];  // for image, label provider
        //        if(error < thresh) {
        //            break;
        //        }
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

// TEST(loader, imagenet)
// {
//     int    height        = 224;
//     int    width         = 224;
//     size_t batch_size    = 128;
//     std::string manifest_root = "/mnt/e/imagenet/I1K_ingested/i1k-extracted/";
//     std::string manifest      = manifest_root + "train-index.tsv";

//     nlohmann::json js = {{"type", "image,label"},
//                             {"manifest_root", manifest_root},
//                             {"manifest_filename", manifest},
//                             {"batch_size", batch_size},
//                             {"iteration_mode", "INFINITE"},
//                             {"cache_directory", "/mnt/c/aeon_cache"},
//                             {"single_thread", false},
//                             {"image",
//                             {{"height", height},
//                             {"width", width},
//                             {"channel_major", false},
//                             {"flip_enable", true}}},
//                             {"label", {{"binary", false}}}};

//     std::chrono::high_resolution_clock timer;
//     std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
//     std::chrono::time_point<std::chrono::high_resolution_clock> zero_time;
//     try
//     {
//         auto train_set = nervana::loader{js};

//         size_t total_batch = ceil((float)train_set.record_count() / (float)batch_size);
//         size_t current_batch = 0;
//         const size_t batches_per_output = 10;
//         for (const nervana::fixed_buffer_map& x : train_set)
//         {
//             (void)x;
//             if (++current_batch % batches_per_output == 0)
//             {
//                 auto last_time = start_time;
//                 start_time = timer.now();
//                 float ms_time =
//                     std::chrono::duration_cast<std::chrono::milliseconds>(start_time - last_time)
//                         .count();
//                 float sec_time = ms_time / 1000.;
//                 std::cout << "batch " << current_batch << " of " << total_batch;
//                 if (last_time != zero_time)
//                 {
//                     std::cout << " time " << ms_time;
//                     std::cout << " " << (float)batches_per_output / sec_time << " batches/s";
//                 }
//                 std::cout << std::endl;
//             }
//         }
//     }
//     catch(std::exception& err)
//     {
//         std::cout << "error processing dataset" << std::endl;
//         std::cout << err.what() << std::endl;
//     }
// }

// TEST(loader, decode_jpeg)
// {
//     stopwatch timer;
//     std::string manifest_root = "/scratch/bob/I1K_ingested/i1k-extracted/";
//     std::string manifest_path = manifest_root + "train-index.tsv";

//     manifest_file manifest{manifest_path, false, manifest_root};

//     vector<vector<string>>* block = nullptr;
//     size_t count = 0;
//     size_t mod = 10000;
//     timer.start();
//     for (block = manifest.next(); block != nullptr; block = manifest.next())
//     {
//         for (const vector<string>& record : *block)
//         {
//             count++;
//             const string& image_path = record[0];
//             cv::Mat image = cv::imread(image_path);
//             if (count % mod == 0)
//             {
//                 auto time = (float)timer.get_milliseconds() / 1000.;
//                 cout << time << endl;
//                 cout << count << " images/second " << ((float)count/time) << endl;
//             }
//         }
//     }
// }

// TEST(loader, load_jpeg)
// {
//     stopwatch timer;
//     std::string manifest_root = "/scratch/bob/I1K_ingested/i1k-extracted/";
//     std::string manifest_path = manifest_root + "train-index.tsv";

//     manifest_file manifest{manifest_path, false, manifest_root};

//     vector<vector<string>>* block = nullptr;
//     size_t count = 0;
//     size_t mod = 10000;
//     timer.start();
//     for (block = manifest.next(); block != nullptr; block = manifest.next())
//     {
//         for (const vector<string>& record : *block)
//         {
//             count++;
//             const string& image_path = record[0];
//             auto data = file_util::read_file_contents(image_path);
//             if (count % mod == 0)
//             {
//                 auto time = (float)timer.get_milliseconds() / 1000.;
//                 cout << count << " images/second " << ((float)count/time) << endl;
//             }
//         }
//     }
// }

// TEST(loader, load_block_manager)
// {
//     stopwatch timer;
//     std::string manifest_root = "/scratch/bob/I1K_ingested/i1k-extracted/";
//     std::string manifest_path = manifest_root + "train-index.tsv";
//     string cache_directory = "/scratch/bob/aeon_cache";
//     bool shuffle = false;
//     size_t block_size = 5000;

//     manifest_file manifest{manifest_path, false, manifest_root};

//     block_loader_file loader{&manifest, block_size};

//     block_manager manager{&loader, 5000, cache_directory, shuffle};

//     encoded_record_list* records;
//     timer.start();
//     float count = 0;
//     while((records = manager.next()) != nullptr)
//     {
//         timer.stop();
//         count = records->size();
//         float time = timer.get_milliseconds()/1000.;
//         cout << "images/second " << count / time << endl;
//         timer.start();
//     }
// }
