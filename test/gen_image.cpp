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

#include <iostream>
#include <sys/stat.h>

#include "gen_image.hpp"
#include "util.hpp"

using namespace std;

gen_image::gen_image()
    : m_image_rows(256)
    , m_image_cols(256)
{
}

gen_image::~gen_image()
{
}

gen_image& gen_image::ImageSize(int rows, int cols)
{
    assert(rows > 0);
    assert(cols > 0);
    m_image_rows = rows;
    m_image_cols = cols;
    return *this;
}

vector<unsigned char> gen_image::render_datum(int number)
{
    cv::Mat image      = cv::Mat(m_image_rows, m_image_cols, CV_8UC3);
    image              = cv::Scalar(255, 255, 255);
    auto     fontFace  = cv::FONT_HERSHEY_PLAIN;
    string   text      = to_string(number); // + ", " + to_string(label);
    float    scale     = 2.0 / 256. * m_image_rows;
    int      thickness = 1;
    int      baseline  = 0;
    cv::Size textSize  = getTextSize(text, fontFace, scale, thickness, &baseline);
    baseline += thickness;

    cv::Point position((m_image_rows - textSize.width) / 2, (m_image_cols + textSize.height) / 2);

    cv::putText(image, text, position, fontFace, scale, cv::Scalar(0, 0, 255));
    vector<unsigned char> result;
    cv::imencode(".png", image, result);
    return result;
}

vector<unsigned char> gen_image::render_target(int number)
{
    int                   target = number + 42;
    vector<unsigned char> rc(4);
    nervana::pack<int>((char*)&rc[0], target);
    return rc;
}

cv::Mat embedded_id_image::generate_image(int rows, int cols, int embedded_id)
{
    cv::Mat  image{rows, cols, CV_8UC3};
    uint8_t* p = image.data;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            *p++ = uint8_t(embedded_id >> 16);
            *p++ = uint8_t(embedded_id >> 8);
            *p++ = uint8_t(embedded_id >> 0);
        }
    }
    return image;
}

int embedded_id_image::read_embedded_id(const cv::Mat& image)
{
    uint8_t* p = image.data;
    int      id;
    id = int(*p++ << 16);
    id |= int(*p++ << 8);
    id |= int(*p++ << 0);
    return id;
}
