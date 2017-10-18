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

#include <map>
#include <string>
#include <memory>

#include "provider_interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_audio.hpp"
#include "etl_blob.hpp"
#include "etl_boundingbox.hpp"
#include "etl_char_map.hpp"
#include "etl_depthmap.hpp"
#include "etl_label_map.hpp"
#include "etl_localization_rcnn.hpp"
#include "etl_localization_ssd.hpp"
#include "etl_pixel_mask.hpp"
#include "etl_video.hpp"
#include "augment_image.hpp"

namespace nervana
{
    namespace provider
    {
        class interface;
        class provider_base;
        class image;
        class label;
        class audio;
        namespace localization
        {
            class rcnn;
            class ssd;
        }
        class pixelmask;
        class boundingbox;
        class blob;
        class video;
        class char_map;
        class label_map;
        class multicrop;
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

    void provide(int idx, encoded_record_list& in_buf, fixed_buffer_map& out_buf) const override;

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
    std::shared_ptr<augment::audio::params> m_audio_augmentations;
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
                 nervana::encoded_record_list& in_buf,
                 nervana::fixed_buffer_map&    out_buf) const
    {
    }
};

//=================================================================================================
// image
//=================================================================================================

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

//=================================================================================================
// audio
//=================================================================================================

class nervana::provider::audio : public provider::interface
{
public:
    audio(nlohmann::json js, nlohmann::json aug);
    virtual ~audio() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::audio::config        m_config;
    nervana::audio::extractor     m_extractor;
    nervana::audio::transformer   m_transformer;
    nervana::audio::loader        m_loader;
    augment::audio::param_factory m_augmentation_factory;
    const std::string             m_buffer_name;
    const std::string             m_length_name;
};

//=================================================================================================
// localization::rcnn
//=================================================================================================

class nervana::provider::localization::rcnn : public provider::interface
{
public:
    rcnn(nlohmann::json js, nlohmann::json aug);
    virtual ~rcnn() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::localization::rcnn::config      m_config;
    nervana::augment::image::param_factory   m_augmentation_factory;
    nervana::localization::rcnn::extractor   m_extractor;
    nervana::localization::rcnn::transformer m_transformer;
    nervana::localization::rcnn::loader      m_loader;
    const std::string                        m_bbtargets_buffer_name;
    const std::string                        m_bbtargets_mask_buffer_name;
    const std::string                        m_labels_flat_buffer_name;
    const std::string                        m_labels_mask_buffer_name;
    const std::string                        m_image_shape_buffer_name;
    const std::string                        m_gt_boxes_buffer_name;
    const std::string                        m_gt_box_count_buffer_name;
    const std::string                        m_gt_class_count_buffer_name;
    const std::string                        m_image_scale_buffer_name;
    const std::string                        m_difficult_flag_buffer_name;
};

//=================================================================================================
// localization::ssd
//=================================================================================================

class nervana::provider::localization::ssd : public provider::interface
{
public:
    ssd(nlohmann::json js, nlohmann::json aug);
    virtual ~ssd() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::localization::ssd::config      m_config;
    nervana::augment::image::param_factory  m_augmentation_factory;
    nervana::localization::ssd::extractor   m_extractor;
    nervana::localization::ssd::transformer m_transformer;
    nervana::localization::ssd::loader      m_loader;
    const std::string                       m_image_shape_buffer_name;
    const std::string                       m_gt_boxes_buffer_name;
    const std::string                       m_gt_box_count_buffer_name;
    const std::string                       m_gt_class_count_buffer_name;
    const std::string                       m_difficult_flag_buffer_name;
};

//=================================================================================================
// pixelmask
//=================================================================================================

class nervana::provider::pixelmask : public provider::interface
{
public:
    pixelmask(nlohmann::json js, nlohmann::json aug);
    virtual ~pixelmask() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::image::config                 m_config;
    nervana::pixel_mask::extractor         m_extractor;
    nervana::pixel_mask::transformer       m_transformer;
    nervana::augment::image::param_factory m_augmentation_factory;
    nervana::image::loader                 m_loader;
    const std::string                      m_buffer_name;
};

//=================================================================================================
// boundingbox
//=================================================================================================

class nervana::provider::boundingbox : public provider::interface
{
public:
    boundingbox(nlohmann::json js, nlohmann::json aug);
    virtual ~boundingbox() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    boundingbox() = delete;
    nervana::boundingbox::config           m_config;
    nervana::boundingbox::extractor        m_extractor;
    nervana::boundingbox::transformer      m_transformer;
    nervana::boundingbox::loader           m_loader;
    nervana::augment::image::param_factory m_augmentation_factory;
    const std::string                      m_buffer_name;
};

//=================================================================================================
// blob
//=================================================================================================

class nervana::provider::blob : public provider::interface
{
public:
    blob(nlohmann::json js);
    virtual ~blob() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    blob() = delete;
    nervana::blob::config    m_config;
    nervana::blob::extractor m_extractor;
    nervana::blob::loader    m_loader;
    const std::string        m_buffer_name;
};

//=================================================================================================
// video
//=================================================================================================

class nervana::provider::video : public provider::interface
{
public:
    video(nlohmann::json js, nlohmann::json aug);
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    nervana::video::config        m_config;
    nervana::video::extractor     m_extractor;
    nervana::video::transformer   m_transformer;
    nervana::video::loader        m_loader;
    augment::image::param_factory m_augmentation_factory;
    const std::string             m_buffer_name;
};

//=================================================================================================
// char_map
//=================================================================================================

class nervana::provider::char_map : public provider::interface
{
public:
    char_map(nlohmann::json js);
    virtual ~char_map() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    char_map() = delete;
    nervana::char_map::config    m_config;
    nervana::char_map::extractor m_extractor;
    nervana::char_map::loader    m_loader;
    const std::string            m_buffer_name;
    const std::string            m_length_name;
};

//=================================================================================================
// label_map
//=================================================================================================

class nervana::provider::label_map : public provider::interface
{
public:
    label_map(nlohmann::json js);
    virtual ~label_map() {}
    void provide(int                        idx,
                 const std::vector<char>&   datum_in,
                 nervana::fixed_buffer_map& out_buf,
                 augmentation&) const override;

private:
    label_map() = delete;
    nervana::label_map::config    m_config;
    nervana::label_map::extractor m_extractor;
    nervana::label_map::loader    m_loader;
    const std::string             m_buffer_name;
};

//=================================================================================================
// multicrop
//=================================================================================================
