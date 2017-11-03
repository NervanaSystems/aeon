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

#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace nervana
{
    namespace image
    {
        // These functions may be common across different transformers
        void resize(const cv::Mat&, cv::Mat&, const cv::Size2i&, bool interpolate = true);
        void expand(const cv::Mat& input, cv::Mat& output, cv::Size offset, cv::Size size);
        void rotate(const cv::Mat&    input,
                    cv::Mat&          output,
                    int               angle,
                    bool              interpolate = true,
                    const cv::Scalar& border      = cv::Scalar());
        void convert_mix_channels(std::vector<cv::Mat>& source,
                                  std::vector<cv::Mat>& target,
                                  std::vector<int>&     from_to);

        void add_padding(cv::Mat& input, int padding, cv::Size2i crop_offset);

        float calculate_scale(const cv::Size& size, int output_width, int output_height);

        cv::Size2f cropbox_max_proportional(const cv::Size2f& in_size, const cv::Size2f& out_size);
        cv::Size2f cropbox_linear_scale(const cv::Size2f& in_size, float scale);
        cv::Size2f cropbox_area_scale(const cv::Size2f& in_size,
                                      const cv::Size2f& cropbox_size,
                                      float             scale);
        cv::Point2f cropbox_shift(const cv::Size2f&, const cv::Size2f&, float, float);

        class photometric
        {
        public:
            photometric();
            static void lighting(cv::Mat& inout, std::vector<float>, float color_noise_std);
            static void cbsjitter(
                cv::Mat& inout, float contrast, float brightness, float saturation, int hue = 0);
            static void transform_hsv(cv::Mat&    image,
                                      const float h_gain,
                                      const float s_gain,
                                      const float v_gain);

            // These are the eigenvectors of the pixelwise covariance matrix
            static const float   _CPCA[3][3];
            static const cv::Mat CPCA;

            // These are the square roots of the eigenvalues of the pixelwise covariance matrix
            static const cv::Mat CSTD;
        };
    }
}
