/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <sstream>

#include "provider.hpp"

using namespace std;
using namespace nervana;

provider::provider_base::provider_base(nlohmann::json                     js,
                                       const std::vector<nlohmann::json>& etl,
                                       nlohmann::json                     augmentation)
    : provider_interface(js, etl.size())
{
    for (nlohmann::json j : etl)
    {
        string type;
        auto   val = j.find("type");
        if (val != j.end())
        {
            type = val->get<string>();
            j.erase(val);
        }
        else
        {
            throw invalid_argument("missing required 'type' element in etl object");
        }
        shared_ptr<provider::interface> prov = nullptr;
        if (type == "image")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::image>(j, augmentation));
        }
        else if (type == "label")
        {
            prov = static_pointer_cast<provider::interface>(make_shared<provider::label>(j));
        }
        else if (type == "audio")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::audio>(j, augmentation));
        }
        else if (type == "localization_rcnn")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::localization::rcnn>(j, augmentation));
        }
        else if (type == "localization_ssd")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::localization::ssd>(j, augmentation));
        }
        else if (type == "pixelmask")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::pixelmask>(j, augmentation));
        }
        else if (type == "boundingbox")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::boundingbox>(j, augmentation));
        }
        else if (type == "blob")
        {
            prov = static_pointer_cast<provider::interface>(make_shared<provider::blob>(j));
        }
        else if (type == "video")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::video>(j, augmentation));
        }
        else if (type == "char_map")
        {
            prov = static_pointer_cast<provider::interface>(make_shared<provider::char_map>(j));
        }
        else if (type == "label_map")
        {
            prov = static_pointer_cast<provider::interface>(make_shared<provider::label_map>(j));
        }
        // else if (type == "multicrop")
        // {
        //     prov = static_pointer_cast<provider::interface>(
        //         make_shared<provider::multicrop>(j, augmentation));
        // }
        else
        {
            stringstream ss;
            ss << "unsupported etl type '" << type << "'";
            throw invalid_argument(ss.str());
        }
        if (prov)
        {
            m_providers.push_back(prov);
            auto os = prov->get_output_shapes();
            m_output_shapes.insert(m_output_shapes.end(), os.begin(), os.end());
        }
    }
}

void provider::provider_base::provide(int                           idx,
                                      nervana::encoded_record_list& in_buf,
                                      nervana::fixed_buffer_map&    out_buf) const
{
    augmentation aug;
    int          index = 0;
    for (const shared_ptr<provider::interface>& provider : m_providers)
    {
        provider->provide(idx, in_buf.record(idx).element(index++), out_buf, aug);
    }
}

//=================================================================================================
// provider::interface
//=================================================================================================

provider::interface::interface(nlohmann::json js, size_t input_count)
    : provider_interface{js, input_count}
{
}

string provider::interface::create_name(const string& name, const string& base_name)
{
    string rc;
    if (name.size() > 0)
    {
        rc = name + ".";
    }
    rc += base_name;
    return rc;
}

//=================================================================================================
// image
//=================================================================================================

provider::image::image(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_transformer{m_config}
    , m_augmentation_factory{aug}
    , m_loader{m_config, m_augmentation_factory.fixed_aspect_ratio}
    , m_buffer_name{create_name(m_config.name, "image")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::image::provide(int                        idx,
                              const std::vector<char>&   datum_in,
                              nervana::fixed_buffer_map& out_buf,
                              augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process image data
    auto decoded    = m_extractor.extract(datum_in.data(), datum_in.size());
    auto input_size = decoded->get_image_size();
    if (aug.m_image_augmentations == nullptr)
    {
        aug.m_image_augmentations = m_augmentation_factory.make_params(
            input_size.width, input_size.height, m_config.width, m_config.height);
    }
    m_loader.load({datum_out}, m_transformer.transform(aug.m_image_augmentations, decoded));
}

//=================================================================================================
// label
//=================================================================================================

provider::label::label(nlohmann::json js)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_loader{m_config}
    , m_buffer_name{create_name(m_config.name, "label")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::label::provide(int                        idx,
                              const vector<char>&        datum_in,
                              nervana::fixed_buffer_map& out_buf,
                              augmentation&              aug) const
{
    char* target_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto label_dec = m_extractor.extract(datum_in.data(), datum_in.size());
    m_loader.load({target_out}, label_dec);
}

//=================================================================================================
// audio
//=================================================================================================

provider::audio::audio(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{}
    , m_transformer{m_config}
    , m_loader{m_config}
    , m_augmentation_factory{aug}
    , m_buffer_name{create_name(m_config.name, "audio")}
    , m_length_name{create_name(m_config.name, "audio_length")}
{
    auto os = m_config.get_shape_type_list();
    m_output_shapes.emplace_back(make_pair(m_buffer_name, os[0]));
    if (m_config.emit_length)
    {
        m_output_shapes.emplace_back(make_pair(m_length_name, os[1]));
    }
}

void provider::audio::provide(int                        idx,
                              const std::vector<char>&   datum_in,
                              nervana::fixed_buffer_map& out_buf,
                              augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    // Process audio data
    auto decoded = m_extractor.extract(datum_in.data(), datum_in.size());
    shared_ptr<augment::audio::params> params;
    if (aug.m_audio_augmentations)
    {
        params = aug.m_audio_augmentations;
    }
    else
    {
        params                    = m_augmentation_factory.make_params();
        aug.m_audio_augmentations = params;
    }
    auto transformed = m_transformer.transform(params, decoded);
    if (m_config.emit_length)
    {
        char* length_out = out_buf[m_length_name]->get_item(idx);
        m_loader.load({datum_out, length_out}, transformed);
    }
    else
    {
        m_loader.load({datum_out}, transformed);
    }
}

//=================================================================================================
// localization::rcnn
//=================================================================================================

provider::localization::rcnn::rcnn(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_augmentation_factory{aug}
    , m_extractor{m_config}
    , m_transformer{m_config, m_augmentation_factory.fixed_scaling_factor}
    , m_loader{m_config}
    , m_bbtargets_buffer_name{create_name(m_config.name, "bbtargets")}
    , m_bbtargets_mask_buffer_name{create_name(m_config.name, "bbtargets_mask")}
    , m_labels_flat_buffer_name{create_name(m_config.name, "labels_flat")}
    , m_labels_mask_buffer_name{create_name(m_config.name, "labels_mask")}
    , m_image_shape_buffer_name{create_name(m_config.name, "image_shape")}
    , m_gt_boxes_buffer_name{create_name(m_config.name, "gt_boxes")}
    , m_gt_box_count_buffer_name{create_name(m_config.name, "gt_box_count")}
    , m_gt_class_count_buffer_name{create_name(m_config.name, "gt_class_count")}
    , m_image_scale_buffer_name{create_name(m_config.name, "image_scale")}
    , m_difficult_flag_buffer_name{create_name(m_config.name, "difficult_flag")}
{
    auto os = m_config.get_shape_type_list();
    m_output_shapes.emplace_back(make_pair(m_bbtargets_buffer_name, os[0]));
    m_output_shapes.emplace_back(make_pair(m_bbtargets_mask_buffer_name, os[1]));
    m_output_shapes.emplace_back(make_pair(m_labels_flat_buffer_name, os[2]));
    m_output_shapes.emplace_back(make_pair(m_labels_mask_buffer_name, os[3]));
    m_output_shapes.emplace_back(make_pair(m_image_shape_buffer_name, os[4]));
    m_output_shapes.emplace_back(make_pair(m_gt_boxes_buffer_name, os[5]));
    m_output_shapes.emplace_back(make_pair(m_gt_box_count_buffer_name, os[6]));
    m_output_shapes.emplace_back(make_pair(m_gt_class_count_buffer_name, os[7]));
    m_output_shapes.emplace_back(make_pair(m_image_scale_buffer_name, os[8]));
    m_output_shapes.emplace_back(make_pair(m_difficult_flag_buffer_name, os[9]));
}

void provider::localization::rcnn::provide(int                        idx,
                                           const std::vector<char>&   datum_in,
                                           nervana::fixed_buffer_map& out_buf,
                                           augmentation&              aug) const
{
    vector<void*> output_list = {out_buf[m_bbtargets_buffer_name]->get_item(idx),
                                 out_buf[m_bbtargets_mask_buffer_name]->get_item(idx),
                                 out_buf[m_labels_flat_buffer_name]->get_item(idx),
                                 out_buf[m_labels_mask_buffer_name]->get_item(idx),
                                 out_buf[m_image_shape_buffer_name]->get_item(idx),
                                 out_buf[m_gt_boxes_buffer_name]->get_item(idx),
                                 out_buf[m_gt_box_count_buffer_name]->get_item(idx),
                                 out_buf[m_gt_class_count_buffer_name]->get_item(idx),
                                 out_buf[m_image_scale_buffer_name]->get_item(idx),
                                 out_buf[m_difficult_flag_buffer_name]->get_item(idx)};

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received localization_rcnn data with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded = m_extractor.extract(datum_in.data(), datum_in.size());
    if (decoded)
    {
        if (aug.m_image_augmentations == nullptr)
        {
            auto input_size           = decoded->input_image_size;
            aug.m_image_augmentations = m_augmentation_factory.make_params(
                input_size.width, input_size.height, m_config.width, m_config.height);
        }
        m_loader.load(output_list, m_transformer.transform(aug.m_image_augmentations, decoded));
    }
}

//=================================================================================================
// localization::ssd
//=================================================================================================

provider::localization::ssd::ssd(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_augmentation_factory{aug}
    , m_extractor{m_config}
    , m_transformer{}
    , m_loader{m_config}
    , m_image_shape_buffer_name{create_name(m_config.name, "image_shape")}
    , m_gt_boxes_buffer_name{create_name(m_config.name, "gt_boxes")}
    , m_gt_box_count_buffer_name{create_name(m_config.name, "gt_box_count")}
    , m_gt_class_count_buffer_name{create_name(m_config.name, "gt_class_count")}
    , m_difficult_flag_buffer_name{create_name(m_config.name, "difficult_flag")}
{
    auto os = m_config.get_shape_type_list();
    m_output_shapes.emplace_back(make_pair(m_image_shape_buffer_name, os[0]));
    m_output_shapes.emplace_back(make_pair(m_gt_boxes_buffer_name, os[1]));
    m_output_shapes.emplace_back(make_pair(m_gt_box_count_buffer_name, os[2]));
    m_output_shapes.emplace_back(make_pair(m_gt_class_count_buffer_name, os[3]));
    m_output_shapes.emplace_back(make_pair(m_difficult_flag_buffer_name, os[4]));
}

void provider::localization::ssd::provide(int                        idx,
                                          const std::vector<char>&   datum_in,
                                          nervana::fixed_buffer_map& out_buf,
                                          augmentation&              aug) const
{
    vector<void*> output_list = {out_buf[m_image_shape_buffer_name]->get_item(idx),
                                 out_buf[m_gt_boxes_buffer_name]->get_item(idx),
                                 out_buf[m_gt_box_count_buffer_name]->get_item(idx),
                                 out_buf[m_gt_class_count_buffer_name]->get_item(idx),
                                 out_buf[m_difficult_flag_buffer_name]->get_item(idx)};

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received localization_ssd data with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    std::shared_ptr<nervana::localization::ssd::decoded> decoded =
        m_extractor.extract(datum_in.data(), datum_in.size());
    if (decoded)
    {
        if (aug.m_image_augmentations == nullptr)
        {
            auto input_size           = decoded->input_image_size;
            aug.m_image_augmentations = m_augmentation_factory.make_ssd_params(input_size.width,
                                                                               input_size.height,
                                                                               m_config.width,
                                                                               m_config.height,
                                                                               decoded->boxes());
        }
        m_loader.load(output_list, m_transformer.transform(aug.m_image_augmentations, decoded));
    }
}

//=================================================================================================
// pixelmask
//=================================================================================================

provider::pixelmask::pixelmask(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_transformer{m_config}
    , m_augmentation_factory{aug}
    , m_loader{m_config, m_augmentation_factory.fixed_aspect_ratio}
    , m_buffer_name{create_name(m_config.name, "pixelmask")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::pixelmask::provide(int                        idx,
                                  const std::vector<char>&   datum_in,
                                  nervana::fixed_buffer_map& out_buf,
                                  augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received pixelmask with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded    = m_extractor.extract(datum_in.data(), datum_in.size());
    auto input_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params;
    if (aug.m_image_augmentations)
    {
        params = aug.m_image_augmentations;
    }
    else
    {
        params = m_augmentation_factory.make_params(
            input_size.width, input_size.height, m_config.width, m_config.height);
        aug.m_image_augmentations = params;
    }
    m_loader.load({datum_out}, m_transformer.transform(params, decoded));
}

//=================================================================================================
// boundingbox
//=================================================================================================

provider::boundingbox::boundingbox(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config.label_map}
    , m_transformer{m_config}
    , m_loader{m_config}
    , m_augmentation_factory{aug}
    , m_buffer_name{create_name(m_config.name, "boundingbox")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::boundingbox::provide(int                        idx,
                                    const std::vector<char>&   datum_in,
                                    nervana::fixed_buffer_map& out_buf,
                                    augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received boundingbox with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded    = m_extractor.extract(datum_in.data(), datum_in.size());
    auto input_size = decoded->image_size();
    shared_ptr<augment::image::params> params;
    if (aug.m_image_augmentations)
    {
        params = aug.m_image_augmentations;
    }
    else
    {
        params = m_augmentation_factory.make_params(
            input_size.width, input_size.height, m_config.width, m_config.height);
        aug.m_image_augmentations = params;
    }
    m_loader.load({datum_out}, m_transformer.transform(params, decoded));
}

//=================================================================================================
// blob
//=================================================================================================

provider::blob::blob(nlohmann::json js)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_loader{m_config}
    , m_buffer_name{create_name(m_config.name, "blob")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::blob::provide(int                        idx,
                             const std::vector<char>&   datum_in,
                             nervana::fixed_buffer_map& out_buf,
                             augmentation&) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received blob with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded = m_extractor.extract(datum_in.data(), datum_in.size());
    m_loader.load({datum_out}, decoded);
}

//=================================================================================================
// video
//=================================================================================================

provider::video::video(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config(js)
    , m_extractor(m_config)
    , m_transformer(m_config)
    , m_loader(m_config)
    , m_augmentation_factory(aug["frame"])
    , m_buffer_name{create_name(m_config.name, "video")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::video::provide(int                        idx,
                              const std::vector<char>&   datum_in,
                              nervana::fixed_buffer_map& out_buf,
                              augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded video with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded    = m_extractor.extract(datum_in.data(), datum_in.size());
    auto image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params;
    if (aug.m_image_augmentations)
    {
        params = aug.m_image_augmentations;
    }
    else
    {
        params = m_augmentation_factory.make_params(
            image_size.width, image_size.height, m_config.frame.width, m_config.frame.height);
        aug.m_image_augmentations = params;
    }
    m_loader.load({datum_out}, m_transformer.transform(params, decoded));
}

//=================================================================================================
// char_map
//=================================================================================================

provider::char_map::char_map(nlohmann::json js)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_loader{m_config}
    , m_buffer_name{create_name(m_config.name, "char_map")}
    , m_length_name{create_name(m_config.name, "char_map_length")}
{
    auto os = m_config.get_shape_type_list();
    m_output_shapes.emplace_back(make_pair(m_buffer_name, os[0]));
    if (m_config.emit_length)
    {
        m_output_shapes.emplace_back(make_pair(m_length_name, os[1]));
    }
}

void provider::char_map::provide(int                        idx,
                                 const std::vector<char>&   datum_in,
                                 nervana::fixed_buffer_map& out_buf,
                                 augmentation&) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received char_map with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    size_t datum_in_size = wstring_length(string(datum_in.data(), datum_in.size()));
    auto   decoded       = m_extractor.extract(datum_in.data(), datum_in_size);
    if (m_config.emit_length)
    {
        char* length_out = out_buf[m_length_name]->get_item(idx);
        m_loader.load({datum_out, length_out}, decoded);
    }
    else
    {
        m_loader.load({datum_out}, decoded);
    }
}

//=================================================================================================
// label_map
//=================================================================================================

provider::label_map::label_map(nlohmann::json js)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_loader{m_config}
    , m_buffer_name{create_name(m_config.name, "label_map")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::label_map::provide(int                        idx,
                                  const std::vector<char>&   datum_in,
                                  nervana::fixed_buffer_map& out_buf,
                                  augmentation&) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received label_map with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto decoded = m_extractor.extract(datum_in.data(), datum_in.size());
    m_loader.load({datum_out}, decoded);
}

//=================================================================================================
// multicrop
//=================================================================================================
