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

#include <algorithm>
#include "augment_image.hpp"
#include "image.hpp"

using namespace std;
using namespace nervana;

using nlohmann::json;
using bbox = boundingbox::box;
using nbox = normalized_box::box;

augment::image::param_factory::param_factory(nlohmann::json js)
{
    if (js.is_null() == false)
    {
        string type;
        auto   val = js.find("type");
        if (val == js.end())
        {
            throw std::invalid_argument("augmentation missing 'type'");
        }
        else
        {
            type = val->get<string>();
            js.erase(val);
        }

        if (type == "image")
        {
            for (auto& info : config_list)
            {
                info->parse(js);
            }
            // verify_config("augment image", config_list, js);

            for (const json& batch_sampler_json : batch_samplers)
            {
                if (batch_sampler_json.is_null())
                {
                    m_batch_samplers.emplace_back();
                }
                else
                {
                    m_batch_samplers.emplace_back(batch_sampler_json);
                }
            }
            if (crop_enable && !m_batch_samplers.empty())
            {
                throw std::invalid_argument(
                    "'Cannot use 'batch_samplers' with 'crop_enable'. Please use only one cropping "
                    "method in augmentations.");
            }

            // Now fill in derived
            if (flip_enable)
            {
                flip_distribution = bernoulli_distribution{0.5};
            }

            if (!center)
            {
                crop_offset = uniform_real_distribution<float>{0.0f, 1.0f};
            }

            if (padding > 0)
            {
                padding_crop_offset_distribution =
                    std::uniform_int_distribution<int>(0, padding * 2);
            }
        }
    }
    m_emit_type = get_emit_constraint_type();
}

emit_type augment::image::param_factory::get_emit_constraint_type()
{
    std::transform(m_emit_constraint_type.begin(),
                   m_emit_constraint_type.end(),
                   m_emit_constraint_type.begin(),
                   ::tolower);
    if (m_emit_constraint_type == "center")
        return emit_type::center;
    else if (m_emit_constraint_type == "min_overlap")
        return emit_type::min_overlap;
    else if (m_emit_constraint_type == "")
        return emit_type::undefined;
    else
        throw std::invalid_argument("Invalid emit constraint type");
}

shared_ptr<augment::image::params> augment::image::param_factory::make_params(
    size_t input_width, size_t input_height, size_t output_width, size_t output_height) const
{
    // Must use this method for creating a shared_ptr rather than make_shared
    // since the params default ctor is private and factory is friend
    // make_shared is not friend :(
    auto settings = shared_ptr<augment::image::params>(new augment::image::params());

    auto& random = get_thread_local_random_engine();

    settings->output_size = cv::Size2i(output_width, output_height);

    settings->angle                  = angle(random);
    settings->flip                   = flip_distribution(random);
    settings->hue                    = hue(random);
    settings->contrast               = contrast(random);
    settings->brightness             = brightness(random);
    settings->saturation             = saturation(random);
    settings->padding                = padding;
    settings->debug_output_directory = debug_output_directory;

    cv::Size2f input_size = cv::Size(input_width, input_height);

    if (!crop_enable)
    {
        int c_off_x                   = padding_crop_offset_distribution(random);
        int c_off_y                   = padding_crop_offset_distribution(random);
        settings->padding_crop_offset = cv::Size2i(c_off_x, c_off_y);
        settings->cropbox             = cv::Rect(cv::Point2f(0, 0), input_size);

        float image_scale;
        if (fixed_scaling_factor > 0)
        {
            image_scale = fixed_scaling_factor;
        }
        else
        {
            image_scale = nervana::image::calculate_scale(input_size, output_width, output_height);
        }
        input_size                   = input_size * image_scale;
        settings->output_size.width  = nervana::unbiased_round(input_size.width);
        settings->output_size.height = nervana::unbiased_round(input_size.height);
    }
    else
    {
        if (padding > 0)
        {
            throw std::invalid_argument(
                "crop_enable should not be true: when padding is defined, crop is executed by "
                "default with cropbox size equal to intput image size");
        }
        float      image_scale            = scale(random);
        float      _horizontal_distortion = horizontal_distortion(random);
        cv::Size2f out_shape(output_width * _horizontal_distortion, output_height);

        cv::Size2f cropbox_size = nervana::image::cropbox_max_proportional(input_size, out_shape);
        if (do_area_scale)
        {
            cropbox_size =
                nervana::image::cropbox_area_scale(input_size, cropbox_size, image_scale);
        }
        else
        {
            cropbox_size = nervana::image::cropbox_linear_scale(cropbox_size, image_scale);
        }

        float c_off_x = crop_offset(random);
        float c_off_y = crop_offset(random);

        cv::Point2f cropbox_origin =
            nervana::image::cropbox_shift(input_size, cropbox_size, c_off_x, c_off_y);
        settings->cropbox = cv::Rect(cropbox_origin, cropbox_size);
    }

    if (lighting.stddev() != 0)
    {
        for (int i = 0; i < 3; i++)
        {
            settings->lighting.push_back(lighting(random));
        }
        settings->color_noise_std = lighting.stddev();
    }

    return settings;
}

shared_ptr<augment::image::params>
    augment::image::param_factory::make_ssd_params(size_t                   input_width,
                                                   size_t                   input_height,
                                                   size_t                   output_width,
                                                   size_t                   output_height,
                                                   const std::vector<bbox>& object_bboxes) const
{
    auto& random   = get_thread_local_random_engine();
    auto  settings = make_params(input_width, input_height, output_width, output_height);
    settings->emit_min_overlap     = m_emit_constraint_min_overlap;
    settings->emit_constraint_type = m_emit_type;

    // use warping
    settings->output_size = cv::Size2i(output_width, output_height);

    // expand
    settings->expand_ratio = expand_ratio(random);
    bool expand_enabled    = expand_distribution(random) < expand_probability;

    cv::Size2f input_size = cv::Size(input_width, input_height);

    if (settings->expand_ratio < 1.)
    {
        throw std::invalid_argument("Expand ratio must be greater than 1.");
    }
    if (expand_enabled)
    {
        float expand_width    = settings->expand_ratio * input_width;
        float expand_height   = settings->expand_ratio * input_height;
        settings->expand_size = cv::Size2i(floor(expand_width), floor(expand_height));

        float max_width_offset  = expand_width - input_width;
        float max_height_offset = expand_height - input_height;
        float w_off             = expand_distribution(random) * max_width_offset;
        float h_off             = expand_distribution(random) * max_height_offset;
        settings->expand_offset = cv::Size2i(floor(w_off), floor(h_off));
    }
    else
    {
        settings->expand_ratio  = 1.0;
        settings->expand_offset = cv::Size2i(0, 0);
        settings->expand_size   = input_size;
    }

    vector<bbox> expanded_object_bboxes(object_bboxes.size());
    vector<nbox> normalized_object_bboxes(object_bboxes.size());
    for (int i = 0; i < object_bboxes.size(); i++)
    {
        try
        {
            expanded_object_bboxes[i] = object_bboxes[i].expand(
                settings->expand_offset, settings->expand_size, settings->expand_ratio);
        }
        catch (exception&)
        {
            ERR << "Failed to expand boxes in make_ssd_params";
            throw;
        }
    }
    try
    {
        normalized_object_bboxes = normalize_bboxes(
            expanded_object_bboxes, settings->expand_size.width, settings->expand_size.height);
    }
    catch (exception&)
    {
        ERR << "Cannot normalize boxes in make_ssd_params";
        throw;
    }
    if (!crop_enable)
    {
        nbox patch = sample_patch(normalized_object_bboxes);
        bbox patch_bbox =
            patch.unnormalize(settings->expand_size.width, settings->expand_size.height);
        settings->cropbox = patch_bbox.rect();
    }

    return settings;
}

nbox augment::image::param_factory::sample_patch(const vector<nbox>& object_bboxes) const
{
    vector<nbox> batch_samples;

    for (const batch_sampler& sampler : m_batch_samplers)
    {
        sampler.sample_patches(object_bboxes, batch_samples);
    }

    if (batch_samples.empty())
    {
        return nbox(0, 0, 1, 1);
    }

    std::uniform_int_distribution<int> uniform_dist(0, batch_samples.size() - 1);
    int                                rand_index  = uniform_dist(get_thread_local_random_engine());
    nbox                               chosen_bbox = batch_samples[rand_index];

    return chosen_bbox;
}

augment::image::sampler::sampler(const nlohmann::json& config)
{
    if (config.is_null())
    {
        return;
    }

    for (auto& info : config_list)
    {
        info->parse(config);
    }
}

void augment::image::sampler::operator=(const nlohmann::json& config)
{
    if (config.is_null())
    {
        return;
    }

    for (auto& info : config_list)
    {
        info->parse(config);
    }
}

nbox augment::image::sampler::sample_patch() const
{
    auto& random           = get_thread_local_random_engine();
    float scale            = m_scale_generator(random);
    float min_aspect_ratio = std::max<float>(m_aspect_ratio_generator.min(), std::pow(scale, 2.));
    float max_aspect_ratio =
        std::min<float>(m_aspect_ratio_generator.max(), 1 / std::pow(scale, 2.));
    auto local_aspect_ratio_generator =
        std::uniform_real_distribution<float>(min_aspect_ratio, max_aspect_ratio);
    float aspect_ratio = local_aspect_ratio_generator(random);

    // Figure out nbox dimension.
    float bbox_width  = scale * sqrt(aspect_ratio);
    float bbox_height = scale / sqrt(aspect_ratio);

    // Figure out top left coordinates.
    float                                 w_off, h_off;
    std::uniform_real_distribution<float> width_generator(0.f, 1.f - bbox_width);
    std::uniform_real_distribution<float> height_generator(0.f, 1.f - bbox_height);
    w_off = width_generator(random);
    h_off = height_generator(random);

    try
    {
        return nbox(w_off, h_off, w_off + bbox_width, h_off + bbox_height);
    }
    catch (exception&)
    {
        ERR << "Error when sampling image:" << endl
            << " scale range: " << m_scale_generator << " scale: " << scale << endl
            << " aspect_ratio range: " << m_aspect_ratio_generator
            << " aspect_ratio: " << aspect_ratio << endl
            << " bbox_width: " << bbox_width << " bbox_height: " << bbox_height << endl
            << " width range: " << width_generator << " w_off: " << w_off << endl
            << " height range: " << height_generator << " h_off: " << h_off;
        throw;
    }
}

bool augment::image::sample_constraint::satisfies(const nbox&              sampled_bbox,
                                                  const std::vector<nbox>& object_bboxes) const
{
    bool has_jaccard_overlap = has_min_jaccard_overlap() || has_max_jaccard_overlap();
    bool has_sample_coverage = has_min_sample_coverage() || has_max_sample_coverage();
    bool has_object_coverage = has_min_object_coverage() || has_max_object_coverage();
    bool satisfy             = !has_jaccard_overlap && !has_sample_coverage && !has_object_coverage;
    if (satisfy)
    {
        // By default, the sampled_bbox is "positive" if no constraints are defined.
        return true;
    }
    // Check constraints.
    bool found = false;
    for (int i = 0; i < object_bboxes.size(); ++i)
    {
        const nbox& object_bbox = object_bboxes[i];
        // Test jaccard overlap.
        if (has_jaccard_overlap)
        {
            const float jaccard_overlap = sampled_bbox.jaccard_overlap(object_bbox);
            if (has_min_jaccard_overlap() && jaccard_overlap < get_min_jaccard_overlap())
            {
                continue;
            }
            if (has_max_jaccard_overlap() && jaccard_overlap > get_max_jaccard_overlap())
            {
                continue;
            }
            found = true;
        }
        // Test sample coverage.
        if (has_sample_coverage)
        {
            const float sample_coverage = sampled_bbox.coverage(object_bbox);
            if (has_min_sample_coverage() && sample_coverage < get_min_sample_coverage())
            {
                continue;
            }
            if (has_max_sample_coverage() && sample_coverage > get_max_sample_coverage())
            {
                continue;
            }
            found = true;
        }
        // Test object coverage.
        if (has_object_coverage)
        {
            const float object_coverage = object_bbox.coverage(sampled_bbox);
            if (has_min_object_coverage() && object_coverage < get_min_object_coverage())
            {
                continue;
            }
            if (has_max_object_coverage() && object_coverage > get_max_object_coverage())
            {
                continue;
            }
            found = true;
        }
        if (found)
        {
            return true;
        }
    }
    return found;
}

augment::image::sample_constraint::sample_constraint(const nlohmann::json& config)
{
    if (config.is_null())
    {
        return;
    }

    for (auto& info : config_list)
    {
        info->parse(config);
    }
}

void augment::image::sample_constraint::operator=(const nlohmann::json& config)
{
    if (config.is_null())
    {
        return;
    }

    for (auto& info : config_list)
    {
        info->parse(config);
    }
}

float augment::image::sample_constraint::get_min_jaccard_overlap() const
{
    if (!has_min_jaccard_overlap())
    {
        throw std::runtime_error("min_jaccard_overlap is not set");
    }
    return m_min_jaccard_overlap;
}

float augment::image::sample_constraint::get_max_jaccard_overlap() const
{
    if (!has_max_jaccard_overlap())
    {
        throw std::runtime_error("max_jaccard_overlap is not set");
    }
    return m_max_jaccard_overlap;
}

float augment::image::sample_constraint::get_min_sample_coverage() const
{
    if (!has_min_sample_coverage())
    {
        throw std::runtime_error("min_sample_coverage is not set");
    }
    return m_min_sample_coverage;
}

float augment::image::sample_constraint::get_max_sample_coverage() const
{
    if (!has_max_sample_coverage())
    {
        throw std::runtime_error("max_sample_coverage is not set");
    }
    return m_max_sample_coverage;
}

float augment::image::sample_constraint::get_min_object_coverage() const
{
    if (!has_min_object_coverage())
    {
        throw std::runtime_error("min_object_coverage is not set");
    }
    return m_min_object_coverage;
}

float augment::image::sample_constraint::get_max_object_coverage() const
{
    if (!has_max_object_coverage())
    {
        throw std::runtime_error("max_object_coverage is not set");
    }
    return m_max_object_coverage;
}

augment::image::batch_sampler::batch_sampler(const nlohmann::json& config)
{
    if (config.is_null())
    {
        return;
    }

    for (auto& info : config_list)
    {
        info->parse(config);
    }

    if (!m_sampler_json.is_null())
    {
        m_sampler = m_sampler_json;
    }

    if (!m_sample_constraint_json.is_null())
    {
        m_sample_constraint = m_sample_constraint_json;
    }
}

void augment::image::batch_sampler::sample_patches(const vector<nbox>& object_bboxes,
                                                   vector<nbox>&       output) const
{
    int found = 0;
    for (int i = 0; i < m_max_trials; ++i)
    {
        if (has_max_sample() && found >= m_max_sample)
        {
            break;
        }
        // Generate sampled_bbox in the normalized space [0, 1].
        nbox sampled_bbox = m_sampler.sample_patch();
        // Determine if the sampled nbox is positive or negative by the constraint.
        if (m_sample_constraint.satisfies(sampled_bbox, object_bboxes))
        {
            ++found;
            output.push_back(sampled_bbox);
        }
    }
}
