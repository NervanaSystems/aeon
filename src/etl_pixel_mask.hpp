/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include "interface.hpp"
#include "etl_image.hpp"
#include "util.hpp"

namespace nervana
{
    namespace pixel_mask
    {
        class extractor;
        class transformer;
        class loader;
    }
}

//-------------------------------------------------------------------------
// Extract
//-------------------------------------------------------------------------

class nervana::pixel_mask::extractor : public interface::extractor<image::decoded>
{
public:
    extractor(const image::config&);
    virtual ~extractor();
    virtual std::shared_ptr<image::decoded> extract(const void*, size_t) const override;

private:
};

//-------------------------------------------------------------------------
// Transform
//-------------------------------------------------------------------------

class nervana::pixel_mask::transformer
    : public interface::transformer<image::decoded, augment::image::params>
{
public:
    transformer(const image::config&);
    ~transformer();
    std::shared_ptr<image::decoded> transform(std::shared_ptr<augment::image::params> txs,
                                              std::shared_ptr<image::decoded> mp) const override;
};

//-------------------------------------------------------------------------
// Load
//-------------------------------------------------------------------------

// use image::loader
