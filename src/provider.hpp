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

#pragma once

#include <map>
#include <string>
#include <memory>

#include "provider_interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "augment_image.hpp"

namespace nervana
{
    namespace provider
    {
        class interface;
        class provider_base;
        class image;
        class dummy_image;
        class label;
    }
    class augmentation;
}

//=================================================================================================
// provider_base
//=================================================================================================

class nervana::provider::provider_base : public provider_interface
{
public:
    provider_base(nlohmann::json                     js,
                  const std::vector<nlohmann::json>& etl,
                  nlohmann::json                     augmentation);

    void provide(int idx, encoded_record& in_buf, fixed_buffer_map& out_buf) const override;

private:
    std::vector<std::shared_ptr<provider::interface>> m_providers;
};

//=================================================================================================
// augmentation
//=================================================================================================

class nervana::augmentation
{
public:
    std::shared_ptr<augment::image::params> m_image_augmentations;
};

//=================================================================================================
// provider_interface
//=================================================================================================

class nervana::provider::interface : public nervana::provider_interface
{
public:
    interface(nlohmann::json, size_t);
    virtual ~interface() {}
    virtual void provide(int                        idx,
                         const std::vector<char>&   datum_in,
                         nervana::fixed_buffer_map& out_buf,
                         augmentation&) const = 0;

    static std::string create_name(const std::string& name, const std::string& base_name);

private:
    void provide(int                           idx,
                 nervana::encoded_record& in_buf,
                 nervana::fixed_buffer_map&    out_buf) const
    {
    }
};

//=================================================================================================
// image
//=================================================================================================
#ifdef WITH_OPENCV
class nervana::provider::image : public provider::interface
{
public:
    image(nlohmann::json config, nlohmann::json aug);
    virtual ~image() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    const nervana::image::config           m_config;
    nervana::image::extractor              m_extractor;
    nervana::image::transformer            m_transformer;
    nervana::augment::image::param_factory m_augmentation_factory;
    nervana::image::loader                 m_loader;
    const std::string                      m_buffer_name;
};
#endif
class nervana::provider::dummy_image : public provider::interface
{
public:
    dummy_image(nlohmann::json config, nlohmann::json aug);
    virtual ~dummy_image() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    const nervana::image::config           m_config;
    nervana::image::dummy_loader           m_loader;
    const std::string                      m_buffer_name;
};

//=================================================================================================
// label
//=================================================================================================

class nervana::provider::label : public provider::interface
{
public:
    label(nlohmann::json config);
    virtual ~label() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::label::config    m_config;
    nervana::label::extractor m_extractor;
    nervana::label::loader    m_loader;
    const std::string         m_buffer_name;
};

