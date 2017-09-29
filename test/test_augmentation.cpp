/*
 Copyright 2017 Nervana Systems Inc.
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

#define private public

#include "augment_image.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

using bbox = boundingbox::box;

void test_sampler(float aspect, float scale, string what = "min_jaccard_overlap", float mv = 0.1)
{
    nlohmann::json batch_samplers = {
        {{"max_sample", 1},
         {"max_trials", 50},
         {"sampler", {{"scale", {scale, scale}}, {"aspect_ratio", {aspect, aspect}}}},
         {"sample_constraint", {{what.c_str(), mv}}}}};

    nlohmann::json js = {
        {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", false}};

    augment::image::param_factory factory(js);

    vector<normalized_box::box> object_bboxes;
    object_bboxes.emplace_back(0.2, 0.2, 0.6, 0.4);
    object_bboxes.emplace_back(0, 0, 0.4, 0.4);
    object_bboxes.emplace_back(0.5, 0.5, 0.6, 0.6);
    object_bboxes.emplace_back(0.2, 0.2, 0.8, 0.6);
    object_bboxes.emplace_back(0.1, 0.0, 0.9, 1.0);
    object_bboxes.emplace_back(0.0, 0.1, 1.0, 0.9);
    object_bboxes.emplace_back(0.0, 0.0, 1.0, 1.0);
    object_bboxes.emplace_back(0.0, 0.0, 1.0, 1.0);
    object_bboxes.emplace_back(0.345, 0.345, 0.35, 0.35);
    object_bboxes.emplace_back(0.9, 0.9, 0.91, 0.91);
    object_bboxes.emplace_back(0.1, 0.9, 0.15, 0.95);
    object_bboxes.emplace_back(0.9, 0.1, 0.95, 0.15);
    object_bboxes.emplace_back(0.56, 0.17, 0.41 + 0.56, 0.17 + 0.59);
    uint32_t non_full_samples_count = 0;
    for (int i = 0; i < 50; i++)
    {
        normalized_box::box out = factory.sample_patch(object_bboxes);
        ASSERT_GE(out.xmin(), 0);
        ASSERT_GE(out.ymin(), 0);
        ASSERT_LE(out.xmax(), 1);
        ASSERT_LE(out.ymax(), 1);
        if (normalized_box::box(0, 0, 1, 1) != out)
        {
            EXPECT_FLOAT_EQ(out.width() / out.height(), aspect);
            EXPECT_GT(out.width(), 0);
            non_full_samples_count++;
        }
    }
    if (!(almost_equal(aspect, 1) && almost_equal(scale, 1)))
    {
        ASSERT_TRUE(non_full_samples_count > 0) << "aspect: " << aspect << " scale: " << scale
                                                << " what: " << what << ": " << mv;
    }
}

TEST(image_augmentation, config)
{
    nlohmann::json js = {{"type", "image"},
                         {"angle", {-20, 20}},
                         {"padding", 4},
                         {"scale", {0.2, 0.8}},
                         {"lighting", {0.0, 0.1}},
                         {"horizontal_distortion", {0.75, 1.33}},
                         {"flip_enable", false}};

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
}

TEST(image_augmnetation, config_crop_and_batch_sampler)
{
    nlohmann::json batch_samplers = {{}};
    nlohmann::json js             = {
        {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", true}};

    EXPECT_THROW(augment::image::param_factory factory(js), std::invalid_argument);
}

TEST(image_augmentation, config_batch_sampler)
{
    nlohmann::json batch_samplers = {
        {{"max_sample", 10},
         {"max_trials", 50},
         {"sampler", {{"scale", {0.1, 1.0}}, {"aspect_ratio", {0.5, 0.7}}}},
         {"sample_constraint",
          {{"min_jaccard_overlap", 0.2},
           {"max_jaccard_overlap", 0.3},
           {"min_sample_coverage", 0.4},
           {"max_sample_coverage", 0.5},
           {"min_object_coverage", 0.6},
           {"max_object_coverage", 0.7}}}},
        {}};

    nlohmann::json js = {
        {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", false}};

    augment::image::param_factory config(js);

    EXPECT_EQ(config.m_batch_samplers.size(), 2);

    const auto& batch_sampler1 = config.m_batch_samplers[0];
    EXPECT_EQ(batch_sampler1.m_max_sample, 10);
    EXPECT_EQ(batch_sampler1.m_max_trials, 50);

    const auto& sampler1 = batch_sampler1.m_sampler;
    EXPECT_FLOAT_EQ(sampler1.m_scale_generator.a(), 0.1);
    EXPECT_FLOAT_EQ(sampler1.m_scale_generator.b(), 1.0);
    EXPECT_FLOAT_EQ(sampler1.m_aspect_ratio_generator.a(), 0.5);
    EXPECT_FLOAT_EQ(sampler1.m_aspect_ratio_generator.b(), 0.7);

    const auto& sample_constraint1 = batch_sampler1.m_sample_constraint;
    EXPECT_TRUE(sample_constraint1.has_min_jaccard_overlap());
    EXPECT_FLOAT_EQ(sample_constraint1.get_min_jaccard_overlap(), 0.2);
    EXPECT_TRUE(sample_constraint1.has_max_jaccard_overlap());
    EXPECT_FLOAT_EQ(sample_constraint1.get_max_jaccard_overlap(), 0.3);
    EXPECT_TRUE(sample_constraint1.has_min_sample_coverage());
    EXPECT_FLOAT_EQ(sample_constraint1.get_min_sample_coverage(), 0.4);
    EXPECT_TRUE(sample_constraint1.has_max_sample_coverage());
    EXPECT_FLOAT_EQ(sample_constraint1.get_max_sample_coverage(), 0.5);
    EXPECT_TRUE(sample_constraint1.has_min_object_coverage());
    EXPECT_FLOAT_EQ(sample_constraint1.get_min_object_coverage(), 0.6);
    EXPECT_TRUE(sample_constraint1.has_max_object_coverage());
    EXPECT_FLOAT_EQ(sample_constraint1.get_max_object_coverage(), 0.7);

    //test default values
    const auto& batch_sampler2 = config.m_batch_samplers[1];
    EXPECT_EQ(batch_sampler2.m_max_sample, -1);
    EXPECT_EQ(batch_sampler2.m_max_trials, 100);

    const auto& sampler2 = batch_sampler2.m_sampler;
    EXPECT_FLOAT_EQ(sampler2.m_scale_generator.a(), 1.0);
    EXPECT_FLOAT_EQ(sampler2.m_scale_generator.b(), 1.0);
    EXPECT_FLOAT_EQ(sampler2.m_aspect_ratio_generator.a(), 1.0);
    EXPECT_FLOAT_EQ(sampler2.m_aspect_ratio_generator.b(), 1.0);

    const auto& sample_constraint2 = batch_sampler2.m_sample_constraint;
    EXPECT_FALSE(sample_constraint2.has_min_jaccard_overlap());
    ASSERT_THROW(sample_constraint2.get_min_jaccard_overlap(), std::runtime_error);
    EXPECT_FALSE(sample_constraint2.has_max_jaccard_overlap());
    ASSERT_THROW(sample_constraint2.get_max_jaccard_overlap(), std::runtime_error);
    EXPECT_FALSE(sample_constraint2.has_min_sample_coverage());
    ASSERT_THROW(sample_constraint2.get_min_sample_coverage(), std::runtime_error);
    EXPECT_FALSE(sample_constraint2.has_max_sample_coverage());
    ASSERT_THROW(sample_constraint2.get_max_sample_coverage(), std::runtime_error);
    EXPECT_FALSE(sample_constraint2.has_min_object_coverage());
    ASSERT_THROW(sample_constraint2.get_min_object_coverage(), std::runtime_error);
    EXPECT_FALSE(sample_constraint2.has_max_object_coverage());
    ASSERT_THROW(sample_constraint2.get_max_object_coverage(), std::runtime_error);
}

TEST(image_augmentation, config_expand_emit_min_overlap)
{
    nlohmann::json js = {{"type", "image"},
                         {"expand_ratio", {2, 4}},
                         {"emit_constraint_type", "min_overlap"},
                         {"emit_constraint_min_overlap", 0.5}};

    augment::image::param_factory config(js);

    EXPECT_FLOAT_EQ(0, config.expand_probability);

    EXPECT_FLOAT_EQ(0, config.expand_distribution.a());
    EXPECT_FLOAT_EQ(1, config.expand_distribution.b());

    EXPECT_FLOAT_EQ(2, config.expand_ratio.a());
    EXPECT_FLOAT_EQ(4, config.expand_ratio.b());

    EXPECT_FLOAT_EQ(0.5, config.m_emit_constraint_min_overlap);
    EXPECT_EQ("min_overlap", config.m_emit_constraint_type);
}

TEST(image_augmentation, config_emit_center)
{
    nlohmann::json js = {{"type", "image"}, {"emit_constraint_type", "center"}};

    augment::image::param_factory config(js);

    EXPECT_EQ("center", config.m_emit_constraint_type);
}

TEST(image_augmentation, config_emit_default)
{
    nlohmann::json js = {{"type", "image"}};

    augment::image::param_factory config(js);

    EXPECT_EQ("", config.m_emit_constraint_type);
}

TEST(image_augmentation, make_ssd_params_default)
{
    int input_width   = 100;
    int input_height  = 100;
    int output_width  = 200;
    int output_height = 300;

    nlohmann::json                     js = {{"type", "image"}, {"crop_enable", false}};
    augment::image::param_factory      factory(js);
    shared_ptr<augment::image::params> params = factory.make_ssd_params(
        input_width, input_height, output_width, output_height, vector<bbox>());

    EXPECT_EQ(params->cropbox, cv::Rect(0, 0, 100, 100));
    EXPECT_EQ(params->expand_offset, cv::Size2i(0, 0));
    EXPECT_EQ(params->expand_size, cv::Size2i(100, 100));
    EXPECT_EQ(params->expand_ratio, 1.0f);
    EXPECT_EQ(params->emit_constraint_type, emit_type::undefined);
    EXPECT_EQ(params->emit_min_overlap, 0.f);
    EXPECT_EQ(params->flip, false);
    EXPECT_EQ(params->angle, 0);
    EXPECT_EQ(params->hue, 0);
    EXPECT_EQ(params->lighting, vector<float>());
    EXPECT_EQ(params->color_noise_std, 0);
    EXPECT_EQ(params->contrast, 1.0);
    EXPECT_EQ(params->brightness, 1.0);
    EXPECT_EQ(params->saturation, 1.0);
    EXPECT_EQ(params->output_size, cv::Size2i(200, 300));
    EXPECT_EQ(params->debug_deterministic, false);
}

TEST(image_augmentation, make_ssd_params_transformations)
{
    int input_width   = 10;
    int input_height  = 10;
    int output_width  = 20;
    int output_height = 30;

    nlohmann::json batch_samplers = {{
        {"max_sample", 1},
        {"max_trials", 50},
        {"sampler", {{"scale", {0.5, 0.5}}, {"aspect_ratio", {1.0, 1.0}}}},
    }};
    nlohmann::json aug = {{"type", "image"},
                          {"batch_samplers", batch_samplers},
                          {"crop_enable", false},
                          {"expand_ratio", {4.0f, 4.0f}},
                          {"expand_probability", 1.0f}};
    augment::image::param_factory      factory(aug);
    auto                               bboxes = vector<bbox>{bbox(0.f, 0.f, 1.f, 1.f)};
    shared_ptr<augment::image::params> params =
        factory.make_ssd_params(input_width, input_height, output_width, output_height, bboxes);

    EXPECT_EQ(params->cropbox.width, 20);
    EXPECT_EQ(params->cropbox.height, 20);
    EXPECT_LE(params->expand_offset.width, 30);
    EXPECT_LE(params->expand_offset.height, 30);
    EXPECT_EQ(params->expand_size, cv::Size2i(40, 40));
    EXPECT_EQ(params->expand_ratio, 4.0f);
    EXPECT_EQ(params->emit_constraint_type, emit_type::undefined);
    EXPECT_EQ(params->emit_min_overlap, 0.f);
    EXPECT_EQ(params->flip, false);
    EXPECT_EQ(params->angle, 0);
    EXPECT_EQ(params->hue, 0);
    EXPECT_EQ(params->lighting, vector<float>());
    EXPECT_EQ(params->color_noise_std, 0);
    EXPECT_EQ(params->contrast, 1.0);
    EXPECT_EQ(params->brightness, 1.0);
    EXPECT_EQ(params->saturation, 1.0);
    EXPECT_EQ(params->output_size, cv::Size2i(20, 30));
    EXPECT_EQ(params->debug_deterministic, false);
}

TEST(image_augmentation, batch_sampler_ratio_scale)
{
    test_sampler(1, 0.5);
    test_sampler(1, 1);
    test_sampler(2, 0.5);
    test_sampler(2, 0.2);
    test_sampler(0.4, 0.6);
}

TEST(image_augmentation, batch_sampler_random_sample_constraint)
{
    std::random_device                    rd;
    std::mt19937                          e2(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    vector<string> strings = {"_jaccard_overlap", "_sample_coverage", "_object_coverage"};

    string b;
    for (int i = 0; i < 20; i++)
    {
        float v1 = dist(e2);
        if (v1 > 0.9)
            b = "max";
        else
            b = "min";
        test_sampler(0.7, 0.5, b + strings.at(i % strings.size()), v1);
    }
}

TEST(image_augmentation, max_sample)
{
    float          aspect         = 1;
    float          scale          = 1;
    int            max_sample     = 10;
    nlohmann::json batch_samplers = {
        {{"max_sample", max_sample},
         {"max_trials", 50},
         {"sampler", {{"scale", {scale, scale}}, {"aspect_ratio", {aspect, aspect}}}},
         {"sample_constraint",
          {{"min_jaccard_overlap", 0.2},
           {"max_jaccard_overlap", 0.3},
           {"min_sample_coverage", 0.4},
           {"max_sample_coverage", 0.5},
           {"min_object_coverage", 0.6},
           {"max_object_coverage", 0.7}}}},
        {}};

    nlohmann::json js = {
        {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", false}};

    augment::image::param_factory factory(js);

    vector<normalized_box::box> object_bboxes;
    object_bboxes.emplace_back(0, 0, 1, 1);
    object_bboxes.emplace_back(0, 0, 0.5, 0.5);
    std::vector<normalized_box::box> output;
    factory.m_batch_samplers[0].sample_patches(object_bboxes, output);
    EXPECT_EQ(output.size(), max_sample);
}

TEST(image_augmentation, max_trials)
{
    std::default_random_engine  e;
    float                       aspect = 1;
    float                       scale  = 1;
    vector<normalized_box::box> object_bboxes;
    object_bboxes.emplace_back(0, 0, 1, 1);
    object_bboxes.emplace_back(0, 0, 0.5, 0.5);
    for (int i = 0; i < 10; i++)
    {
        int            max_trials     = 1 + e() % 100;
        nlohmann::json batch_samplers = {
            {{"max_sample", 1000},
             {"max_trials", max_trials},
             {"sampler", {{"scale", {scale, scale}}, {"aspect_ratio", {aspect, aspect}}}},
             {"sample_constraint",
              {{"min_jaccard_overlap", 0.2},
               {"max_jaccard_overlap", 0.3},
               {"min_sample_coverage", 0.4},
               {"max_sample_coverage", 0.5},
               {"min_object_coverage", 0.6},
               {"max_object_coverage", 0.7}}}},
            {}};

        nlohmann::json js = {
            {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", false}};

        augment::image::param_factory factory(js);

        std::vector<normalized_box::box> output;
        factory.m_batch_samplers[0].sample_patches(object_bboxes, output);
        EXPECT_LE(output.size(), max_trials) << "at iteration " << i;
    }
}

TEST(image_augmentation, default_patch)
{
    nlohmann::json batch_samplers = {
        {{"max_sample", 10},
         {"max_trials", 50},
         {"sampler", {{"scale", {0.1, 1}}, {"aspect_ratio", {0.5, 2}}}},
         {"sample_constraint",
          {{"min_jaccard_overlap", 0.3},
           {"max_jaccard_overlap", 0.3},
           {"min_sample_coverage", 0.4},
           {"max_sample_coverage", 0.4},
           {"min_object_coverage", 0.5},
           {"max_object_coverage", 0.5}}}},
    };

    nlohmann::json js = {
        {"type", "image"}, {"batch_samplers", batch_samplers}, {"crop_enable", false}};

    augment::image::param_factory factory(js);

    vector<normalized_box::box> object_bboxes;
    object_bboxes.emplace_back(0, 0, 1, 1);
    object_bboxes.emplace_back(0, 0, 0.5, 0.5);
    std::vector<normalized_box::box> output;
    factory.m_batch_samplers[0].sample_patches(object_bboxes, output);

    for (int i = 0; i < output.size(); i++)
    {
        EXPECT_EQ(output[i], normalized_box::box(0, 0, 1, 1)) << "at iteration " << i;
    }
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
