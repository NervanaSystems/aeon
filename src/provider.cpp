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
#ifdef WITH_OPENCV
        if (type == "image")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::image>(j, augmentation));
        }
        else
#endif
        if (type == "dummy_image")
        {
            prov = static_pointer_cast<provider::interface>(
                make_shared<provider::dummy_image>(j, augmentation));
        }
        else if (type == "label")
        {
            prov = static_pointer_cast<provider::interface>(make_shared<provider::label>(j));
        }
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
                                      nervana::encoded_record& in_buf,
                                      nervana::fixed_buffer_map&    out_buf) const
{
    augmentation aug;
    int          index = 0;
    for (const shared_ptr<provider::interface>& provider : m_providers)
    {
        provider->provide(idx, in_buf.element(index++), out_buf, aug);
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
#ifdef WITH_OPENCV
provider::image::image(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_extractor{m_config}
    , m_transformer{m_config}
    , m_augmentation_factory{aug}
    , m_loader{m_config,
               m_augmentation_factory.fixed_aspect_ratio,
               m_augmentation_factory.mean,
               m_augmentation_factory.stddev}
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
#endif


provider::dummy_image::dummy_image(nlohmann::json js, nlohmann::json aug)
    : interface(js, 1)
    , m_config{js}
    , m_loader{m_config}
    , m_buffer_name{create_name(m_config.name, "dummy_image")}
{
    m_output_shapes.emplace_back(make_pair(m_buffer_name, m_config.get_shape_type()));
}

void provider::dummy_image::provide(int                        idx,
                              const std::vector<char>&   datum_in,
                              nervana::fixed_buffer_map& out_buf,
                              augmentation&              aug) const
{
    char* datum_out = out_buf[m_buffer_name]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded dummy_image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process dummy_imag data
    m_loader.load({datum_out}, {});
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

