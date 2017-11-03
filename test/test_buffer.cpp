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

#include "buffer_batch.hpp"
#include "helpers.hpp"
#include "log.hpp"
#include "file_util.hpp"
#include "provider_factory.hpp"
#include "provider_interface.hpp"

using namespace std;
using namespace nervana;

void read(encoded_record_list& b, const string& str)
{
    encoded_record record;
    record.add_element(string2vector(str));
    b.add_record(record);
}

TEST(buffer, shuffle)
{
    // create a buffer with lots of words in sorted order.  assert
    // that they are sorted, then shuffle, then assert that they are
    // not sorted

    encoded_record_list b;

    read(b, "abc");
    read(b, "asd");
    read(b, "hello");
    read(b, "qwe");
    read(b, "world");
    read(b, "xyz");
    read(b, "yuiop");
    read(b, "zxcvb");

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), true);

    b.shuffle(0);

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), false);
}

void setup_buffer_exception(encoded_record_list& b)
{
    // setup b with length 4, one value is an exception
    read(b, "a");

    try
    {
        throw std::runtime_error("expect me");
    }
    catch (std::exception&)
    {
        encoded_record record;
        record.add_exception(std::current_exception());
        b.add_record(record);
    }

    read(b, "c");
    read(b, "d");
}

TEST(buffer, write_exception)
{
    encoded_record_list b;

    setup_buffer_exception(b);
}

TEST(buffer, read_exception)
{
    encoded_record_list b;

    setup_buffer_exception(b);

    // no exceptions if we hit index 0, 2 and 3
    ASSERT_EQ(b.record(0).element(0)[0], 'a');
    ASSERT_EQ(b.record(2).element(0)[0], 'c');
    ASSERT_EQ(b.record(3).element(0)[0], 'd');

    // assert that exception is raised
    try
    {
        b.record(1).element(0);
        FAIL();
    }
    catch (std::exception& e)
    {
        ASSERT_STREQ("expect me", e.what());
    }
}

TEST(buffer, serialization)
{
    int        height     = 24;
    int        width      = 24;
    size_t     batch_size = 8;
    const bool pinned     = false;

    using nlohmann::json;
    json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label_config = {{"type", "label"}, {"binary", false}};
    json config       = {{"manifest_root", ""},
                   {"manifest_filename", ""},
                   {"batch_size", batch_size},
                   {"iteration_mode", "INFINITE"},
                   {"cache_directory", ""},
                   {"decode_thread_count", 0},
                   {"etl", {image_config, label_config}}};

    shared_ptr<nervana::provider_interface> provider = provider_factory::create(config);

    // generate fixed_buffer_map sample
    nervana::fixed_buffer_map fbm(provider->get_output_shapes(), batch_size, pinned);

    std::minstd_rand0 rand_items(0);
    for (auto name : fbm.get_names())
    {
        for (int i               = 0; i < fbm[name]->size(); i++)
            fbm[name]->data()[i] = rand_items() % 100 + 32;
    }

    // serialize
    stringstream ss;
    ss << fbm;

    // deserialize
    nervana::fixed_buffer_map fbm_restored;
    ss >> fbm_restored;

    // compare buffers
    fbm_restored.get_names();
    ASSERT_EQ(fbm_restored.get_names().size(), fbm.get_names().size());

    for (int i = 0; i < fbm.get_names().size(); i++)
    {
        ASSERT_EQ(fbm_restored.get_names()[i], fbm.get_names()[i]);
    }

    for (auto name : fbm.get_names())
    {
        ASSERT_EQ(fbm[name]->size(), fbm_restored[name]->size());
        EXPECT_TRUE(0 ==
                    std::memcmp(fbm[name]->data(), fbm_restored[name]->data(), fbm[name]->size()));
        EXPECT_TRUE(fbm[name]->get_shape_type() == fbm_restored[name]->get_shape_type());
    }

    // /////////////////////////////////////////////////////////
    stringstream ss_spahes;
    ss_spahes << provider->get_output_shapes();

    std::vector<std::pair<std::string, nervana::shape_type>> shapes_restored;
    ss_spahes >> shapes_restored;
    for (auto shape : provider->get_output_shapes())
    {
        bool found = false;
        for (auto shape_restored : shapes_restored)
        {
            if (std::get<0>(shape_restored) == std::get<0>(shape))
            {
                ASSERT_EQ(std::get<1>(shape), std::get<1>(shape_restored));
                found = true;
            }
        }
        ASSERT_TRUE(found);
    }
}
