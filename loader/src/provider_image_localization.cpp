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

image_localization::image_localization(nlohmann::json js) :
    image_config(js["image"]),
    localization_config(js["localization"], image_config),
    image_extractor(image_config),
    image_transformer(image_config),
    image_loader(image_config),
    image_factory(image_config),
    localization_extractor(localization_config),
    localization_transformer(localization_config),
    localization_loader(localization_config)
{
    num_inputs = 2;
    oshapes.push_back(image_config.get_shape_type());
    auto os = localization_config.get_shape_type_list();
    oshapes.insert(oshapes.end(), os.begin(), os.end());
}

void image_localization::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) {
    vector<char>& datum_in  = in_buf[0]->get_item(idx);
    vector<char>& target_in = in_buf[1]->get_item(idx);

    char* datum_out             = out_buf[0]->get_item(idx);
    char* y_bbtargets_out       = out_buf[1]->get_item(idx);
    char* y_bbtargets_mask_out  = out_buf[2]->get_item(idx);
    char* y_labels_flat_out     = out_buf[3]->get_item(idx);
    char* y_labels_mask_out     = out_buf[4]->get_item(idx);
    char* im_shape_out          = out_buf[5]->get_item(idx);
    char* gt_boxes_out          = out_buf[6]->get_item(idx);
    char* num_gt_boxes_out      = out_buf[7]->get_item(idx);
    char* gt_classes_out        = out_buf[8]->get_item(idx);
    char* im_scale_out          = out_buf[9]->get_item(idx);
    char* gt_difficult          = out_buf[10]->get_item(idx);

    vector<void*> target_list = {
        y_bbtargets_out,
        y_bbtargets_mask_out,
        y_labels_flat_out,
        y_labels_mask_out,
        im_shape_out,
        gt_boxes_out,
        num_gt_boxes_out,
        gt_classes_out,
        im_scale_out,
        gt_difficult
    };

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec = image_extractor.extract(datum_in.data(), datum_in.size());
    if(image_dec) {
        auto image_params = image_factory.make_params(image_dec);
        image_loader.load({datum_out}, image_transformer.transform(image_params, image_dec));

        // Process target data
        auto target_dec = localization_extractor.extract(target_in.data(), target_in.size());
        if(target_dec) {
            localization_loader.load(target_list, localization_transformer.transform(image_params, target_dec));
        }
    }
}
