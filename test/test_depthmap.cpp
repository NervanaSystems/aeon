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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gtest/gtest.h"

#define private public

#include "etl_depthmap.hpp"
#include "json.hpp"
#include "helpers.hpp"

using namespace std;
using namespace nervana;

cv::Mat generate_test_image()
{
    cv::Mat        color = cv::Mat(256, 256, CV_8UC3);
    unsigned char* input = (unsigned char*)(color.data);
    int            index = 0;
    for (int row = 0; row < 256; row++)
    {
        for (int col = 0; col < 256; col++)
        {
            uint8_t value  = ((row + col) % 2 ? 0xFF : 0x00);
            input[index++] = value; // b
            input[index++] = value; // g
            input[index++] = value; // r
        }
    }
    return color;
}

// pixels must be either black or white
bool verify_image(cv::Mat img)
{
    unsigned char* data  = (unsigned char*)(img.data);
    int            index = 0;
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            for (int channel = 0; channel < img.channels(); channel++)
            {
                if (data[index] != 0x00 && data[index] != 0xFF)
                    return false;
                index++;
            }
        }
    }
    return true;
}
