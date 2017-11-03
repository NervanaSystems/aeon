/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#include "raw_image.hpp"

using namespace std;
using namespace nervana;

static cv::Mat generate_indexed_image()
{
    cv::Mat        color = cv::Mat(256, 256, CV_8UC3);
    unsigned char* input = (unsigned char*)(color.data);
    int            index = 0;
    for (int row = 0; row < 256; row++)
    {
        for (int col = 0; col < 256; col++)
        {
            input[index++] = col; // b
            input[index++] = row; // g
            input[index++] = 0;   // r
        }
    }
    return color;
}

TEST(raw_image, write_read)
{
    string file = "raw.bin";

    {
        auto mat = generate_indexed_image();
        auto raw = raw_image::from_cvmat(mat);
        raw.write(file);
    }

    {
        raw_image raw{file};
        auto      mat = raw.to_cvmat();
        cv::imwrite("raw.png", mat);
    }
}
