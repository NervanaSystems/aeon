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

#include <string>
#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "dataset.hpp"

class gen_image : public dataset<gen_image>
{
public:
    gen_image();
    virtual ~gen_image();

    gen_image& ImageSize(int rows, int cols);

private:
    int m_image_rows;
    int m_image_cols;

    std::vector<unsigned char> render_target(int datumNumber) override;
    std::vector<unsigned char> render_datum(int datumNumber) override;

    std::vector<unsigned char> render_image(int number, int label);
};

class embedded_id_image
{
public:
    static cv::Mat generate_image(int rows, int cols, int embedded_id);
    static int read_embedded_id(const cv::Mat& image);
};
