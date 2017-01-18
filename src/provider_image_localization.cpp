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

#include "provider_image_localization.hpp"

using namespace nervana;
using namespace std;

image_localization::image_localization(nlohmann::json js)
    : provider_interface(js, 2)
    , image_config(js["image"])
    , localization_config(js["localization"], image_config)
    , image_extractor(image_config)
    , image_transformer(image_config)
    , image_loader(image_config)
    , image_factory(image_config)
    , localization_extractor(localization_config)
    , localization_transformer(localization_config)
    , localization_loader(localization_config)
{
    m_output_shapes.insert({"image", image_config.get_shape_type()});

    auto os = localization_config.get_shape_type_list();
    m_output_shapes.insert({"bbtargets", os[0]});
    m_output_shapes.insert({"bbtargets_mask", os[1]});
    m_output_shapes.insert({"labels_flat", os[2]});
    m_output_shapes.insert({"labels_mask", os[3]});
    m_output_shapes.insert({"image_shape", os[4]});
    m_output_shapes.insert({"gt_boxes", os[5]});
    m_output_shapes.insert({"gt_box_count", os[6]});
    m_output_shapes.insert({"gt_class_count", os[7]});
    m_output_shapes.insert({"image_scale", os[8]});
    m_output_shapes.insert({"difficult_flag", os[9]});
}

void image_localization::provide(int idx, encoded_record_list& in_buf, fixed_buffer_map& out_buf)
{
    vector<char>& datum_in  = in_buf.record(idx).element(0);
    vector<char>& target_in = in_buf.record(idx).element(1);

    char* datum_out          = out_buf["image"]->get_item(idx);
    char* bbtargets_out      = out_buf["bbtargets"]->get_item(idx);
    char* bbtargets_mask_out = out_buf["bbtargets_mask"]->get_item(idx);
    char* labels_flat_out    = out_buf["labels_flat"]->get_item(idx);
    char* labels_mask_out    = out_buf["labels_mask"]->get_item(idx);
    char* image_shape_out    = out_buf["image_shape"]->get_item(idx);
    char* gt_boxes_out       = out_buf["gt_boxes"]->get_item(idx);
    char* num_gt_boxes_out   = out_buf["gt_box_count"]->get_item(idx);
    char* gt_classes_out     = out_buf["gt_class_count"]->get_item(idx);
    char* image_scale_out    = out_buf["image_scale"]->get_item(idx);
    char* gt_difficult       = out_buf["difficult_flag"]->get_item(idx);

    vector<void*> target_list = {bbtargets_out,
                                 bbtargets_mask_out,
                                 labels_flat_out,
                                 labels_mask_out,
                                 image_shape_out,
                                 gt_boxes_out,
                                 num_gt_boxes_out,
                                 gt_classes_out,
                                 image_scale_out,
                                 gt_difficult};

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    if (image_dec)
    {
        auto image_params = image_factory.make_params(image_dec);
        image_loader.load({datum_out}, image_transformer.transform(image_params, image_dec));

        // Process target data
        auto target_dec = localization_extractor.extract(target_in.data(), target_in.size());
        if (target_dec)
        {
            localization_loader.load(target_list,
                                     localization_transformer.transform(image_params, target_dec));
        }
    }
}
