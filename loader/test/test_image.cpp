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
#include "json.hpp"

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

void test_image(vector<unsigned char>& img, int channels) {
    nlohmann::json js = {{"height",30},{"width",30},{"channels",channels},{
    "distribution",{
        {"angle",{-20,20}},
        {"scale",{0.2,0.8}},
        {"lighting",{0.0,0.1}},
        {"aspect_ratio",{0.75,1.33}},
        {"flip",{false}}
    }}};
    string cfgString = js.dump(4);

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
    EXPECT_EQ(channels,mat.channels());

    // unsigned char *input = (unsigned char*)(mat.data);
    // int index = 0;
    // for(int row = 0; row < 256; row++) {
    //     for(int col = 0; col < 256; col++) {
    //         if(channels == 3) {
    //             EXPECT_EQ(col,input[index++]);
    //             EXPECT_EQ(row,input[index++]);
    //             index++;
    //         } else if(channels == 1) {
    //         }
    //     }
    // }
}

TEST(etl, image_config) {
    nlohmann::json js = {{"height",30},{"width",30},{"channels",3},{
    "distribution",{
        {"angle",{-20,20}},
        {"scale",{0.2,0.8}},
        {"lighting",{0.0,0.1}},
        {"aspect_ratio",{0.75,1.33}},
        {"flip",{false}}
    }}};
    string cfgString = js.dump(4);

    // cout << cfgString << endl;

    auto config = make_shared<image::config>(cfgString);
    EXPECT_EQ(30,config->height);
    EXPECT_EQ(30,config->width);
    EXPECT_FALSE(config->do_area_scale);
    EXPECT_TRUE(config->channel_major);
    EXPECT_EQ(3,config->channels);

    EXPECT_FLOAT_EQ(0.2,config->scale.a());
    EXPECT_FLOAT_EQ(0.8,config->scale.b());

    EXPECT_EQ(-20,config->angle.a());
    EXPECT_EQ(20,config->angle.b());

    EXPECT_FLOAT_EQ(0.0,config->lighting.mean());
    EXPECT_FLOAT_EQ(0.1,config->lighting.stddev());

    EXPECT_FLOAT_EQ(0.75,config->aspect_ratio.a());
    EXPECT_FLOAT_EQ(1.33,config->aspect_ratio.b());

    EXPECT_FLOAT_EQ(0.0,config->photometric.a());
    EXPECT_FLOAT_EQ(0.0,config->photometric.b());

    EXPECT_FLOAT_EQ(0.5,config->crop_offset.a());
    EXPECT_FLOAT_EQ(0.5,config->crop_offset.b());

    EXPECT_FLOAT_EQ(0.0,config->flip.p());
}

TEST(etl, image_extract1) {
    auto indexed = generate_indexed_image();

    vector<unsigned char> png;
    cv::imencode( ".png", indexed, png );

    test_image( png, 3 );
}

TEST(etl, image_extract2) {
    auto indexed = generate_indexed_image();

    vector<unsigned char> png;
    cv::imencode( ".png", indexed, png );

    test_image( png, 1 );
}

TEST(etl, image_extract3) {
    cv::Mat img = cv::Mat( 256, 256, CV_8UC1 );

    vector<unsigned char> png;
    cv::imencode( ".png", img, png );

    test_image( png, 3 );
}

TEST(etl, image_extract4) {
    cv::Mat img = cv::Mat( 256, 256, CV_8UC1 );

    vector<unsigned char> png;
    cv::imencode( ".png", img, png );

    test_image( png, 1 );
}
