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

#include <stdexcept>
#include "gtest/gtest.h"

#include "loader.hpp"

using namespace std;
using namespace nervana;

TEST(config, loader)
{
    int height = 32;
    int width  = 32;

    // config is valid
    nlohmann::json image = {{"type", "image"},
                            {"name", "image1"},
                            {"height", height},
                            {"width", width},
                            {"channel_major", false}};
    nlohmann::json label = {{"type", "label"}, {"name", "label1"}, {"binary", false}};
    nlohmann::json js = {{"manifest_filename", "blah"}, {"batch_size", 1}, {"etl", {image, label}}};

    loader_config cfg{js};
    // EXPECT_NO_THROW(loader_config cfg{js});
}

TEST(config, throws)
{
    // config is missing a required parameter

    int            height        = 32;
    int            width         = 32;
    size_t         batch_size    = 1;
    string         manifest_root = string(CURDIR) + "/test_data";
    string         manifest      = manifest_root + "/manifest.tsv";
    nlohmann::json image         = {{"type", "image"},
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
                         {"type", "image,label"},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image, label}},
                         {"augmentation", augmentation}};

    EXPECT_THROW(loader_config cfg{js}, invalid_argument);
}
