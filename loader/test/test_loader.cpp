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
#include "csv_manifest_maker.hpp"
#include "gen_image.hpp"
using namespace std;
using namespace nervana;

TEST(loader,iteration_mode)
{
    int height = 32;
    int width = 32;
    size_t batch_size = 1;
    string manifest = string(CURDIR)+"/test_data/manifest.csv";

    {
        nlohmann::json js = {{"type","image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", string(CURDIR) + "/test_data"},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"image", {
                                {"height",height},
                                {"width",width},
                                {"channel_major",false},
                                {"flip_enable",true}}},
                             {"label", {
                                  {"binary",false}
                              }
                             }};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type","image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", string(CURDIR) + "/test_data"},
                             {"batch_size", batch_size},
                             {"iteration_mode", "ONCE"},
                             {"image", {
                                {"height",height},
                                {"width",width},
                                {"channel_major",false},
                                {"flip_enable",true}}},
                             {"label", {
                                  {"binary",false}
                              }
                             }};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type","image,label"},
                             {"manifest_filename", manifest},
                             {"manifest_root", string(CURDIR) + "/test_data"},
                             {"batch_size", batch_size},
                             {"iteration_mode", "COUNT"},
                             {"iteration_mode_count", 1000},
                             {"image", {
                                {"height",height},
                                {"width",width},
                                {"channel_major",false},
                                {"flip_enable",true}}},
                             {"label", {
                                  {"binary",false}
                              }
                             }};

        auto train_set = loader{js};
    }

    {
        nlohmann::json js = {{"type","image,label"},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "COUNT"},
                             {"image", {
                                {"height",height},
                                {"width",width},
                                {"channel_major",false},
                                {"flip_enable",true}}},
                             {"label", {
                                  {"binary",true}
                              }
                             }};

        EXPECT_THROW(loader{js}, std::invalid_argument);
    }

    {
        nlohmann::json js = {{"type","image,label"},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "BLAH"},
                             {"image", {
                                {"height",height},
                                {"width",width},
                                {"channel_major",false},
                                {"flip_enable",true}}},
                             {"label", {
                                  {"binary",true}
                              }
                             }};

        EXPECT_THROW(loader{js}, std::invalid_argument);
    }
}

TEST(loader,iterator)
{
    int height = 32;
    int width = 32;
    size_t batch_size = 1;
    size_t input_file_count = 10;
    manifest_maker mm(input_file_count, height, width);
    string manifest_filename = mm.get_manifest_name();
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"image", {
                            {"height",height},
                            {"width",width},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

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
    for (int i=2; i<input_file_count; i++)
    {
        EXPECT_NE(begin, end);
        begin++;
    }
    EXPECT_EQ(input_file_count, begin.position());
    EXPECT_EQ(begin, end);
}

TEST(loader,once)
{
    int height = 32;
    int width = 32;
    size_t batch_size = 1;
    size_t input_file_count = 10;
    manifest_maker mm(input_file_count, height, width);
    string manifest_filename = mm.get_manifest_name();
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "ONCE"},
                         {"image", {
                            {"height",height},
                            {"width",width},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

    loader train_set{js};

    int count = 0;
    for(const fixed_buffer_map& data : train_set)
    {
        (void)data; // silence compiler warning
        ASSERT_NE(count, input_file_count);
        count++;
    }
    ASSERT_EQ(input_file_count, count);
}

TEST(loader,count)
{
    int height = 32;
    int width = 32;
    size_t batch_size = 1;
    size_t input_file_count = 10;
    manifest_maker mm(input_file_count, height, width);
    string manifest_filename = mm.get_manifest_name();
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "COUNT"},
                         {"iteration_mode_count", 4},
                         {"image", {
                            {"height",height},
                            {"width",width},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

    int expected_iterations = 4;

    loader train_set{js};

    int count = 0;
    ASSERT_EQ(expected_iterations, train_set.m_batch_count_value);
    for(const fixed_buffer_map& data : train_set)
    {
        (void)data; // silence compiler warning
        ASSERT_NE(count, input_file_count);
        count++;
    }
    ASSERT_EQ(expected_iterations, count);
}

TEST(loader,infinite)
{
    int height = 32;
    int width = 32;
    size_t batch_size = 1;
    size_t input_file_count = 10;
    manifest_maker mm(input_file_count, height, width);
    string manifest_filename = mm.get_manifest_name();
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"iteration_mode", "INFINITE"},
                         {"image", {
                            {"height",height},
                            {"width",width},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

    loader train_set{js};

    int count = 0;
    int expected_iterations = input_file_count * 3;
    for(const fixed_buffer_map& data : train_set)
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

TEST(loader,test)
{
    int height = 32;
    int width = 64;
    size_t batch_size = 32;
    size_t input_file_count = 1000;
    manifest_maker mm(input_file_count, height, width);
    string manifest_filename = mm.get_manifest_name();
    nlohmann::json js = {{"type","image,label"},
                         {"manifest_filename", manifest_filename},
                         {"batch_size", batch_size},
                         {"image", {
                            {"height",height},
                            {"width",width},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

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

    // Range based first
    // Run pass twice to ensure that breaking and restarting starts from the beginning
    for (int pass_number = 0; pass_number < 2; ++pass_number)
    {
        int count=0;
        int expected_id = 0;
        for(const fixed_buffer_map& data : train_set)  // if d1 created with infinite, this will just keep going
        {
            ASSERT_EQ(2, data.size());

            const buffer_fixed_size_elements* image_buffer_ptr = data["image"];
            ASSERT_NE(nullptr, image_buffer_ptr);
            const buffer_fixed_size_elements& image_buffer = *image_buffer_ptr;
            for(int i=0; i<batch_size; i++)
            {
                const char* image_data = image_buffer.get_item(i);
                cv::Mat image{height, width, CV_8UC3, (char*)image_data};
                int actual_id = embedded_id_image::read_embedded_id(image);
                ASSERT_EQ(expected_id%input_file_count, actual_id);
                expected_id++;
            }

            if(count++ == 8)
            {
                break;
            }
        }
    }

    for (int pass_number = 0; pass_number < 2; ++pass_number)
    {
        train_set.reset();
        int count=0;
        int expected_id = 0;
        loader::iterator& ts_iter = train_set.get_current_iter();

        while (ts_iter != train_set.get_end_iter())
        {
            const fixed_buffer_map& data = *ts_iter;
            ASSERT_EQ(2, data.size());

            const buffer_fixed_size_elements* image_buffer_ptr = data["image"];
            ASSERT_NE(nullptr, image_buffer_ptr);
            const buffer_fixed_size_elements& image_buffer = *image_buffer_ptr;
            for(int i=0; i<batch_size; i++)
            {
                const char* image_data = image_buffer.get_item(i);
                cv::Mat image{height, width, CV_8UC3, (char*)image_data};
                int actual_id = embedded_id_image::read_embedded_id(image);
                ASSERT_EQ(expected_id%input_file_count, actual_id);
                expected_id++;
            }

            ++ts_iter;
            if(count++ == 8)
            {
                break;
            }
        }
    }

//    all_errors = [];

//    for(auto data : valid_set)  // since d2 created with "once"
//    {
//        //all_errors.append(calc_batch_error(data))
//    }

    // now we've accumulated for the entire set:  (maybe a bit too much) Suppose 100 data, and batch_size 75

//    len(all_errors.size()) == 150
//    epoch_errors = all_errors[:len(d2)]

//    valid_set.reset();
//    sleep(2);
}



// TEST(loader,mnist)
// {
//     size_t batch_size = 32;
//     nlohmann::json js = {{"type","image,label"},
//                          {"manifest_filename", "/scratch/alex/mnist/train-index.csv"},
//                          {"batch_size", batch_size},
//                          {"image", {
//                             {"height",28},
//                             {"width",28},
//                             {"channels",1}}},
//                          {"label", {
//                               {"binary",false}
//                           }
//                          }};

//     loader train_set{js};

//     auto buf_names = train_set.get_buffer_names();
//     EXPECT_EQ(2, buf_names.size());
//     EXPECT_NE(find(buf_names.begin(), buf_names.end(), "image"), buf_names.end());
//     EXPECT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

//     auto image_shape = train_set.get_shape("image");
//     auto label_shape = train_set.get_shape("label");

//     int count=0;
//     int expected_id = 0;

//     auto itt = train_set.begin();
//     while (itt != train_set.end())
//     {
//         const fixed_buffer_map& data = *itt;
//         const buffer_fixed_size_elements* label_buffer_ptr = data["label"];
//         ASSERT_NE(nullptr, label_buffer_ptr);
//         const buffer_fixed_size_elements& label_buffer = *label_buffer_ptr;
//         for(int i=0; i<batch_size; i++)
//         {
//             uint32_t *dptr = (uint32_t *) label_buffer.get_item(i);
//             INFO << "train_loop " << *dptr;
//         }
//         ++itt;
//         if(count++ == 8)
//         {
//             break;
//         }
//     }

//     // for(const fixed_buffer_map& data : train_set)
//     // {
//     //     ASSERT_EQ(2, data.size());

//     //     const buffer_fixed_size_elements* label_buffer_ptr = data["label"];
//     //     ASSERT_NE(nullptr, label_buffer_ptr);
//     //     const buffer_fixed_size_elements& label_buffer = *label_buffer_ptr;
//     //     for(int i=0; i<batch_size; i++)
//     //     {
//     //         uint32_t *dptr = (uint32_t *) label_buffer.get_item(i);
//     //         INFO << "train_loop " << *dptr;
//     //     }

//     //     if(count++ == 8)
//     //     {
//     //         break;
//     //     }
//     // }
// }

