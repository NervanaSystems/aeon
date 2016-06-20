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

#pragma once

#include "etl_image.hpp"

namespace nervana {
    namespace video {
        class config;
        class params;
        class decoded;

        // goes from config -> params
        class param_factory;

        class extractor;
        class transformer;
        class loader;
    }

    class video::params : public nervana::params {
    public:
        params() {}
        void dump(std::ostream & = std::cout);

        nervana::image::params _frameParams;
        int _framesPerClip;
    };

    class video::decoded : public decoded_media {
    public:
        decoded() {}
        virtual ~decoded() override {}

        virtual MediaType get_type() override { return MediaType::VIDEO; }
    protected:
        nervana::image::decoded _images;
    };

    class video::extractor : public interface::extractor<video::decoded> {
        extractor(std::shared_ptr<const video::config>);
        ~extractor() {}

        virtual std::shared_ptr<video::decoded> extract(const char*, int) override;
    };
}
