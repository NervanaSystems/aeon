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

#pragma once

#include <iostream>
#include <memory>
#include <limits>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "boundingbox.hpp"
#include "normalized_box.hpp"
#include "json.hpp"
#include "interface.hpp"
#include "box.hpp"

namespace nervana
{
    namespace augment
    {
        namespace image
        {
            class params;
            class param_factory;

            class sample;
            class sampler;
            class sample_constraint;
            class batch_sampler;
        }
    }
}

enum class emit_type
{
    center,
    min_overlap,
    undefined
};

static std::ostream& operator<<(std::ostream& out, const emit_type& et)
{
    switch (et)
    {
    case emit_type::center: out << "center"; break;
    case emit_type::min_overlap: out << "min_overlap"; break;
    case emit_type::undefined: out << "undefined"; break;
    default: throw std::out_of_range("cannot print provided emit_type value");
    }
    return out;
}

class nervana::augment::image::params
{
    friend class nervana::augment::image::param_factory;

public:
    friend std::ostream& operator<<(std::ostream& out, const params& obj)
    {
        out << "expand_ratio         " << obj.expand_ratio << "\n";
        out << "expand_offset        " << obj.expand_offset << "\n";
        out << "expand_size          " << obj.expand_size << "\n";
        out << "emit_constraint_type " << obj.emit_constraint_type << "\n";
        out << "emit_min_overlap     " << obj.emit_min_overlap << "\n";
        out << "cropbox                " << obj.cropbox << "\n";
        out << "output_size            " << obj.output_size << "\n";
        out << "angle                  " << obj.angle << "\n";
        out << "flip                   " << obj.flip << "\n";
        out << "padding                " << obj.padding << "\n";
        out << "padding_crop_offset    " << obj.padding_crop_offset << "\n";
        out << "lighting               " << join(obj.lighting, ", ") << "\n";
        out << "color_noise_std        " << obj.color_noise_std << "\n";
        out << "contrast               " << obj.contrast << "\n";
        out << "brightness             " << obj.brightness << "\n";
        out << "saturation             " << obj.saturation << "\n";
        out << "hue                    " << obj.hue << "\n";
        out << "debug_deterministic    " << obj.debug_deterministic << "\n";
        out << "debug_output_directory " << obj.debug_output_directory << "\n";
        return out;
    }

    float              expand_ratio = 1.0;
    cv::Size2i         expand_offset;
    cv::Size2i         expand_size;
    emit_type          emit_constraint_type = emit_type::undefined;
    float              emit_min_overlap     = 0.f;
    cv::Rect           cropbox;
    cv::Size2i         output_size;
    int                angle = 0;
    bool               flip  = false;
    int                padding;
    cv::Size2i         padding_crop_offset;
    std::vector<float> lighting; // pixelwise random values
    float              color_noise_std        = 0;
    float              contrast               = 1.0;
    float              brightness             = 1.0;
    float              saturation             = 1.0;
    int                hue                    = 0;
    bool               debug_deterministic    = false;
    std::string        debug_output_directory = "";

private:
    params() {}
};

class nervana::augment::image::param_factory : public json_configurable
{
public:
    param_factory(nlohmann::json config);
    std::shared_ptr<params> make_params(size_t input_width,
                                        size_t input_height,
                                        size_t output_width,
                                        size_t output_height) const;

    std::shared_ptr<params>
        make_ssd_params(size_t                                        input_width,
                        size_t                                        input_height,
                        size_t                                        output_width,
                        size_t                                        output_height,
                        const std::vector<nervana::boundingbox::box>& object_bboxes) const;

    bool        do_area_scale                 = false;
    bool        crop_enable                   = true;
    bool        fixed_aspect_ratio            = false;
    float       expand_probability            = 0.;
    float       fixed_scaling_factor          = -1;
    std::string m_emit_constraint_type        = "";
    float       m_emit_constraint_min_overlap = 0.0;

    /** Scale the crop box (width, height) */
    mutable std::uniform_real_distribution<float> scale{1.0f, 1.0f};

    /** Rotate the image (rho, phi) */
    mutable std::uniform_int_distribution<int> angle{0, 0};

    /** Adjust lighting */
    mutable std::normal_distribution<float> lighting{0.0f, 0.0f};

    /** Adjust aspect ratio */
    mutable std::uniform_real_distribution<float> horizontal_distortion{1.0f, 1.0f};

    /** Adjust contrast */
    mutable std::uniform_real_distribution<float> contrast{1.0f, 1.0f};

    /** Adjust brightness */
    mutable std::uniform_real_distribution<float> brightness{1.0f, 1.0f};

    /** Adjust saturation */
    mutable std::uniform_real_distribution<float> saturation{1.0f, 1.0f};

    /** Expand image */
    mutable std::uniform_real_distribution<float> expand_ratio{1.0f, 1.0f};
    mutable std::uniform_real_distribution<float> expand_distribution{0.0f, 1.0f};

    /** Rotate hue in degrees. Valid values are [-180; 180] */
    mutable std::uniform_int_distribution<int> hue{0, 0};

    /** Offset from center for the crop */
    mutable std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};

    /** Flip the image left to right */
    mutable std::bernoulli_distribution flip_distribution{0};

    /** Image padding pixel number with random crop to original image size */
    int padding{0};

    /** Writes transformed data to the provided directory */
    std::string debug_output_directory = "";

    std::vector<nlohmann::json> batch_samplers;

private:
    nervana::normalized_box::box sample_patch(
        const std::vector<nervana::normalized_box::box>& normalized_object_bboxes) const;

    bool      flip_enable = false;
    bool      center      = true;
    emit_type m_emit_type;

    /** Offset for padding cropbox */
    mutable std::uniform_int_distribution<int> padding_crop_offset_distribution{0, 0};

    std::vector<batch_sampler> m_batch_samplers;

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_DISTRIBUTION(scale,
                         mode::OPTIONAL,
                         [](const std::uniform_real_distribution<float>& v) {
                             return v.a() >= 0 && v.a() <= 1 && v.b() >= 0 && v.b() <= 1 &&
                                    v.a() <= v.b();
                         }),
        ADD_DISTRIBUTION(angle, mode::OPTIONAL, [](decltype(angle) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
        ADD_DISTRIBUTION(horizontal_distortion,
                         mode::OPTIONAL,
                         [](decltype(horizontal_distortion) v) { return v.a() <= v.b(); }),
        ADD_SCALAR(flip_enable, mode::OPTIONAL),
        ADD_SCALAR(center, mode::OPTIONAL),
        ADD_SCALAR(do_area_scale, mode::OPTIONAL),
        ADD_SCALAR(crop_enable, mode::OPTIONAL),
        ADD_SCALAR(expand_probability, mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_emit_constraint_type, "emit_constraint_type", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(
            m_emit_constraint_min_overlap, "emit_constraint_min_overlap", mode::OPTIONAL),
        ADD_SCALAR(fixed_aspect_ratio, mode::OPTIONAL),
        ADD_SCALAR(fixed_scaling_factor, mode::OPTIONAL),
        ADD_SCALAR(padding, mode::OPTIONAL),
        ADD_SCALAR(debug_output_directory, mode::OPTIONAL),
        ADD_DISTRIBUTION(
            contrast, mode::OPTIONAL, [](decltype(contrast) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(
            brightness, mode::OPTIONAL, [](decltype(brightness) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(
            saturation, mode::OPTIONAL, [](decltype(saturation) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(expand_ratio,
                         mode::OPTIONAL,
                         [](decltype(expand_ratio) v) { return v.a() >= 1 && v.a() <= v.b(); }),
        ADD_DISTRIBUTION(hue, mode::OPTIONAL, [](decltype(hue) v) { return v.a() <= v.b(); }),
        ADD_OBJECT(batch_samplers, mode::OPTIONAL)};

    emit_type get_emit_constraint_type();
};

class nervana::augment::image::sample final
{
public:
    explicit sample(float scale, float aspect_ratio)
        : m_scale(scale)
        , m_aspect_ratio(aspect_ratio)
    {
    }
    float get_scale() const { return m_scale; }
    float get_aspect_ratio() const { return m_aspect_ratio; }
private:
    float m_scale;
    float m_aspect_ratio;
};

class nervana::augment::image::sampler final : public json_configurable
{
public:
    sampler() = default;
    explicit sampler(const nlohmann::json& config);
    void operator=(const nlohmann::json& config);

    normalized_box::box sample_patch() const;

private:
    /** Scale of sampled box */
    mutable std::uniform_real_distribution<float> m_scale_generator{1.0f, 1.0f};
    /** Aspect Ratio of sampled box */
    mutable std::uniform_real_distribution<float> m_aspect_ratio_generator{1.0f, 1.0f};

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_DISTRIBUTION_WITH_KEY(m_scale_generator,
                                  "scale",
                                  mode::OPTIONAL,
                                  [](decltype(m_scale_generator) v) {
                                      return v.a() <= v.b() && v.a() > 0. && v.b() <= 1.;
                                  }),
        ADD_DISTRIBUTION_WITH_KEY(m_aspect_ratio_generator,
                                  "aspect_ratio",
                                  mode::OPTIONAL,
                                  [](decltype(m_aspect_ratio_generator) v) {
                                      return v.a() <= v.b() && v.a() > 0. && v.b() < FLT_MAX;
                                  })};
};

class nervana::augment::image::sample_constraint final : public json_configurable
{
public:
    sample_constraint() = default;
    explicit sample_constraint(const nlohmann::json& config);
    void operator=(const nlohmann::json& config);

    bool satisfies(const normalized_box::box&              normalized_sampled_bbox,
                   const std::vector<normalized_box::box>& normalized_object_bboxes) const;

    bool  has_min_jaccard_overlap() const { return !std::isnan(m_min_jaccard_overlap); }
    float get_min_jaccard_overlap() const;

    bool  has_max_jaccard_overlap() const { return !std::isnan(m_max_jaccard_overlap); }
    float get_max_jaccard_overlap() const;

    bool  has_min_sample_coverage() const { return !std::isnan(m_min_sample_coverage); }
    float get_min_sample_coverage() const;

    bool  has_max_sample_coverage() const { return !std::isnan(m_max_sample_coverage); }
    float get_max_sample_coverage() const;

    bool  has_min_object_coverage() const { return !std::isnan(m_min_object_coverage); }
    float get_min_object_coverage() const;

    bool  has_max_object_coverage() const { return !std::isnan(m_max_object_coverage); }
    float get_max_object_coverage() const;

private:
    float m_min_jaccard_overlap = std::numeric_limits<float>::quiet_NaN();
    float m_max_jaccard_overlap = std::numeric_limits<float>::quiet_NaN();

    float m_min_sample_coverage = std::numeric_limits<float>::quiet_NaN();
    float m_max_sample_coverage = std::numeric_limits<float>::quiet_NaN();

    float m_min_object_coverage = std::numeric_limits<float>::quiet_NaN();
    float m_max_object_coverage = std::numeric_limits<float>::quiet_NaN();

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR_WITH_KEY(m_min_jaccard_overlap, "min_jaccard_overlap", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_max_jaccard_overlap, "max_jaccard_overlap", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_min_sample_coverage, "min_sample_coverage", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_max_sample_coverage, "max_sample_coverage", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_min_object_coverage, "min_object_coverage", mode::OPTIONAL),
        ADD_SCALAR_WITH_KEY(m_max_object_coverage, "max_object_coverage", mode::OPTIONAL)};
};

class nervana::augment::image::batch_sampler : public json_configurable
{
public:
    batch_sampler() = default;
    batch_sampler(const nlohmann::json& config);

    void sample_patches(const std::vector<normalized_box::box>& normalized_object_bboxes,
                        std::vector<normalized_box::box>&       normalized_output) const;

private:
    // If provided, break when found certain number of samples satisfing the
    // sample_constraint. Value -1 means the value is not provided.
    int m_max_sample = -1;

    // Maximum number of trials for sampling to avoid infinite loop.
    unsigned int m_max_trials = 100;

    // Constraints for sampling bbox.
    nlohmann::json m_sampler_json;

    sampler m_sampler;

    // Constraints for determining if a sampled bbox is positive or negative.
    nlohmann::json m_sample_constraint_json;

    sample_constraint m_sample_constraint;

    bool has_max_sample() const { return m_max_sample != -1; }
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR_WITH_KEY(
            m_max_sample, "max_sample", mode::OPTIONAL, [](int x) { return x >= 0; }),
        ADD_SCALAR_WITH_KEY(m_max_trials, "max_trials", mode::OPTIONAL),
        ADD_JSON(m_sampler_json, "sampler", mode::OPTIONAL),
        ADD_JSON(m_sample_constraint_json, "sample_constraint", mode::OPTIONAL)};
};
