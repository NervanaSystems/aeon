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

#include <iostream>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gen_image.hpp"
#include "util.hpp"

using namespace std;

gen_image::gen_image()
    : _imageRows(256)
    , _imageCols(256)
{
}

gen_image::~gen_image()
{
}

gen_image& gen_image::ImageSize(int rows, int cols)
{
    assert(rows > 0);
    assert(cols > 0);
    _imageRows = rows;
    _imageCols = cols;
    return *this;
}

vector<unsigned char> gen_image::render_datum(int number)
{
    cv::Mat image      = cv::Mat(_imageRows, _imageCols, CV_8UC3);
    image              = cv::Scalar(255, 255, 255);
    auto     fontFace  = cv::FONT_HERSHEY_PLAIN;
    string   text      = to_string(number); // + ", " + to_string(label);
    float    scale     = 2.0 / 256. * _imageRows;
    int      thickness = 1;
    int      baseline  = 0;
    cv::Size textSize  = getTextSize(text, fontFace, scale, thickness, &baseline);
    baseline += thickness;

    cv::Point position((_imageRows - textSize.width) / 2, (_imageCols + textSize.height) / 2);

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
