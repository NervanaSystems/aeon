/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public

#include "augment_image.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

TEST(image_augmentation, config)
{
    nlohmann::json js = {{"type", "image"},
                         {"angle", {-20, 20}},
                         {"padding", 4},
                         {"scale", {0.2, 0.8}},
                         {"lighting", {0.0, 0.1}},
                         {"horizontal_distortion", {0.75, 1.33}},
                         {"flip_enable", false},
                         {"mean", {0.5, 0.4, 0.6}}};

    augment::image::param_factory config(js);
    EXPECT_FALSE(config.do_area_scale);

    EXPECT_FLOAT_EQ(0.2, config.scale.a());
    EXPECT_FLOAT_EQ(0.8, config.scale.b());

    EXPECT_EQ(-20, config.angle.a());
    EXPECT_EQ(20, config.angle.b());

    EXPECT_EQ(4, config.padding);
    EXPECT_EQ(0, config.padding_crop_offset_distribution.a());
    EXPECT_EQ(8, config.padding_crop_offset_distribution.b());

    EXPECT_FLOAT_EQ(0.0, config.lighting.mean());
    EXPECT_FLOAT_EQ(0.1, config.lighting.stddev());

    EXPECT_FLOAT_EQ(0.75, config.horizontal_distortion.a());
    EXPECT_FLOAT_EQ(1.33, config.horizontal_distortion.b());

    EXPECT_FLOAT_EQ(1.0, config.contrast.a());
    EXPECT_FLOAT_EQ(1.0, config.contrast.b());

    EXPECT_FLOAT_EQ(1.0, config.brightness.a());
    EXPECT_FLOAT_EQ(1.0, config.brightness.b());
    EXPECT_FLOAT_EQ(1.0, config.saturation.a());
    EXPECT_FLOAT_EQ(1.0, config.saturation.b());

    EXPECT_FLOAT_EQ(0.5, config.crop_offset.a());
    EXPECT_FLOAT_EQ(0.5, config.crop_offset.b());

    EXPECT_FLOAT_EQ(0.0, config.flip_distribution.p());

    EXPECT_THAT(config.mean, testing::ElementsAre(0.5, 0.4, 0.6));
    EXPECT_TRUE(config.stddev.empty());
}

TEST(image_augmentation, padding_with_crop_enabled)
{
    nlohmann::json js = {{"type", "image"},
                         {"angle", {-20, 20}},
                         {"padding", 4},
                         {"scale", {0.2, 0.8}},
                         {"lighting", {0.0, 0.1}},
                         {"horizontal_distortion", {0.75, 1.33}},
                         {"crop_enable", true},
                         {"flip_enable", true}};

    augment::image::param_factory config(js);
    EXPECT_THROW(config.make_params(10, 10, 10, 10), std::invalid_argument);
}

