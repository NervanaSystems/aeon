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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#define private public

#include "params.hpp"
#include "etl_image_var.hpp"
#include "json.hpp"

using namespace std;
using namespace nervana;

static cv::Mat generate_indexed_image() {
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

TEST(image_var, decoded_image) {
    cv::Mat img1 = cv::Mat( 256, 256, CV_8UC3 );

    image_var::decoded decoded(img1);
}

TEST(image_var, image_config) {
    nlohmann::json js = {{"min_size",300},{"max_size",400},{"channels",3},{
    "distribution",{
        {"flip",{false}}
    }}};

    image_var::config config(js);
    EXPECT_EQ(300,config.min_size);
    EXPECT_EQ(400,config.max_size);
    EXPECT_TRUE(config.channel_major);
    EXPECT_EQ(3,config.channels);

    EXPECT_FLOAT_EQ(0.0,config.flip.p());
}

static bool check_value(shared_ptr<image_var::decoded> transformed, int x0, int y0, int x1, int y1) {
    cv::Mat image = transformed->get_image();
    cv::Vec3b value = image.at<cv::Vec3b>(y0,x0); // row,col
    return x1 == (int)value[0] && y1 == (int)value[1];
}

TEST(image_var, resize) {
    auto mat = cv::Mat(200,300,CV_8UC3);
    vector<unsigned char> img;
    cv::imencode( ".png", mat, img );
    cout << __PRETTY_FUNCTION__ << " mat size " << mat.size() << endl;

    nlohmann::json jsConfig = {{"min_size",300},{"max_size",400},{"channels",3},{
    "distribution",{
        {"flip",{false}}
    }}};

    image_var::config config_ptr{jsConfig};

    image_var::extractor ext{config_ptr};
    shared_ptr<image_var::decoded> decoded = ext.extract((char*)&img[0], img.size());

    image_var::param_factory factory(config_ptr);

    shared_ptr<image_var::params> params_ptr = make_shared<image_var::params>();

    image_var::transformer trans{config_ptr};
    shared_ptr<image_var::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image();
    EXPECT_EQ(400,image.size().width);
    EXPECT_EQ(267,image.size().height);
}

TEST(image_var, transform_flip) {
    auto indexed = generate_indexed_image();
    vector<unsigned char> img;
    cv::imencode( ".png", indexed, img );

    nlohmann::json jsConfig = {{"min_size",256},{"max_size",256},{"channels",3},{
    "distribution",{
        {"flip",{true}}
    }}};

    image_var::config config_ptr{jsConfig};

    image_var::extractor ext{config_ptr};
    shared_ptr<image_var::decoded> decoded = ext.extract((char*)&img[0], img.size());

    std::default_random_engine dre;
    image_var::param_factory factory(config_ptr);

    shared_ptr<image_var::params> params_ptr = make_shared<image_var::params>();
    params_ptr->flip = true;

    image_var::transformer trans{config_ptr};
    shared_ptr<image_var::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image();
    EXPECT_EQ(256,image.size().width);
    EXPECT_EQ(256,image.size().height);

    EXPECT_TRUE(check_value(transformed,0,0,255,0));
    EXPECT_TRUE(check_value(transformed,100,100,255-100,100));
}
