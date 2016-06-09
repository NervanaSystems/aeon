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

#include <vector>
#include <string>
#include <sstream>
#include <random>

#include "gtest/gtest.h"

#include "params.hpp"
#include "etl_image.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nervana;

cv::Mat generate_indexed_image() {
    cv::Mat color = cv::Mat( 256, 256, CV_8UC3 );
    unsigned char *input = (unsigned char*)(color.data);
    int index = 0;
    for(int row = 0; row < 256; row++) {
        for(int col = 0; col < 256; col++) {
            input[index++] = col;       // b
            input[index++] = row;       // g
            input[index++] = 0;         // r
        }
    }
    return color;
}

void test_image(vector<unsigned char>& img, int num_channels) {
    string cfgString = R"(
        {
            "height": 30,
            "width" : 30,
            "num_channels" : )"+to_string(num_channels)+R"(,
            "dist_params/angle" : [-20, 20],
            "dist_params/scale" : [0.2, 0.8],
            "dist_params/lighting" : [0.0, 0.1],
            "dist_params/aspect_ratio" : [0.75, 1.33],
            "dist_params/flip" : [false]
        }
    )";

    cout << cfgString << endl;

    auto itpj = make_shared<image::config>(cfgString);

    nervana::image::extractor ext{itpj};
    std::shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    ASSERT_NE(nullptr,decoded);
    EXPECT_EQ(1,decoded->size());
    cv::Size2i size = decoded->get_image_size(0);
    EXPECT_EQ(256,size.width);
    EXPECT_EQ(256,size.height);
    cv::Mat mat = decoded->get_image(0);
    EXPECT_EQ(256,mat.rows);
    EXPECT_EQ(256,mat.cols);
    EXPECT_EQ(num_channels,mat.channels());

    // unsigned char *input = (unsigned char*)(mat.data);
    // int index = 0;
    // for(int row = 0; row < 256; row++) {
    //     for(int col = 0; col < 256; col++) {
    //         if(num_channels == 3) {
    //             EXPECT_EQ(col,input[index++]);
    //             EXPECT_EQ(row,input[index++]);
    //             index++;
    //         }
    //     }
    // }
}

TEST(etl, image_extract) {
    {
        auto indexed = generate_indexed_image();
        cv::imwrite( "indexed.png", indexed );

        vector<unsigned char> png;
        cv::imencode( ".png", indexed, png );

        test_image( png, 3 );
    }
}
