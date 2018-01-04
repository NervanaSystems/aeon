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
#define NP_NO_DEPRACATED_API NPY_1_7_API_VERSION

#include "python_utils.hpp"
#include "conversion.hpp"

namespace nervana
{
    class plugin
    {
    public:
        std::string get_name() { return filename; }
    private:
        std::string filename;
        PyObject*   handle{nullptr};
        PyObject*   klass{nullptr};
        PyObject*   instance{nullptr};

        template <typename T>
        T augment(const std::string& methodname, const T& in_data);

    public:
        plugin() = delete;
        plugin(const std::string& filename, const std::string& params);

        void    prepare();
        cv::Mat augment_image(const cv::Mat& m);
        std::vector<boundingbox::box>
            augment_boundingbox(const std::vector<boundingbox::box>& boxes);
        cv::Mat augment_audio(const cv::Mat& m);
        cv::Mat augment_pixel_mask(const cv::Mat& m);
        cv::Mat augment_depthmap(const cv::Mat& m);
    };
}
