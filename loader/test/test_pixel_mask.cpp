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

#include "provider_image_class.hpp"
#include "etl_image.hpp"
#include "json.hpp"
#include "helpers.hpp"

using namespace std;
using namespace nervana;

static cv::Mat generate_test_image() {
    cv::Mat color = cv::Mat( 256, 256, CV_8UC3 );
    unsigned char *input = (unsigned char*)(color.data);
    int index = 0;
    for(int row = 0; row < 256; row++) {
        for(int col = 0; col < 256; col++) {
            uint8_t value = ((row+col)%2 ? 0xFF : 0x00);
            input[index++] = value;       // b
            input[index++] = value;       // g
            input[index++] = value;       // r
        }
    }
    return color;
}

// pixels must be either black or white
static bool verify_image(cv::Mat img) {
    unsigned char *data = (unsigned char*)(img.data);
    int index = 0;
    for(int row = 0; row < img.rows; row++) {
        for(int col = 0; col < img.cols; col++) {
            if(data[index] != 0x00 && data[index] != 0xFF) return false;
            index++;
            if(data[index] != 0x00 && data[index] != 0xFF) return false;
            index++;
            if(data[index] != 0x00 && data[index] != 0xFF) return false;
            index++;
        }
    }
    return true;
}

TEST(pixel_mask, scale_up) {
    auto test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256},{"height",256}};
    image::config cfg(js);

    image::extractor        extractor{cfg};
    pixel_mask::transformer transformer{cfg};
    image::loader           loader{cfg};
    image::param_factory    factory{cfg};

    auto extracted = extractor.extract((const char*)test_data.data(), test_data.size());
    image_params_builder builder(factory.make_params(extracted));
    shared_ptr<image::params> params_ptr = builder.output_size(300, 300);
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat tximg = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_scale_up.png",tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, scale_down) {
    auto test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256},{"height",256}};
    image::config cfg(js);

    image::extractor        extractor{cfg};
    pixel_mask::transformer transformer{cfg};
    image::loader           loader{cfg};
    image::param_factory    factory{cfg};

    auto extracted = extractor.extract((const char*)test_data.data(), test_data.size());
    image_params_builder builder(factory.make_params(extracted));
    shared_ptr<image::params> params_ptr = builder.output_size(100, 100);
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat tximg = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_scale_down.png",tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, rotate) {
    auto test_image = generate_test_image();
    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);
    ASSERT_TRUE(verify_image(test_image));

    nlohmann::json js = {{"width", 256},{"height",256}};
    image::config cfg(js);

    image::extractor        extractor{cfg};
    pixel_mask::transformer transformer{cfg};
    image::loader           loader{cfg};
    image::param_factory    factory{cfg};

    auto extracted = extractor.extract((const char*)test_data.data(), test_data.size());
    image_params_builder builder(factory.make_params(extracted));
    shared_ptr<image::params> params_ptr = builder.angle(45);
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat tximg = transformed->get_image(0);
    cv::imwrite("tx_pixel_mask_rotate.png",tximg);
    EXPECT_TRUE(verify_image(tximg));
}

TEST(pixel_mask, load_int) {
    cv::Mat test_image(256, 256, CV_8UC3);
    uint8_t* input = (uint8_t*)(test_image.data);
    int index = 0;
    for(int row = 0; row < 256; row++) {
        for(int col = 0; col < 256; col++) {
            uint8_t value = col;
            input[index++] = value;       // b
            input[index++] = value;       // g
            input[index++] = value;       // r
        }
    }



    vector<uint8_t> test_data;
    cv::imencode(".png", test_image, test_data);

    nlohmann::json js = {
        {"width", 256},
        {"height",256},
        {"type_string", "int32_t"},
        {"channels", 1}
    };
    image::config cfg(js);

    image::extractor        extractor{cfg};
    pixel_mask::transformer transformer{cfg};
    image::loader           loader{cfg};
    image::param_factory    factory{cfg};

    auto extracted = extractor.extract((const char*)test_data.data(), test_data.size());
    image_params_builder builder(factory.make_params(extracted));
    shared_ptr<image::params> params_ptr = builder;
    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
    cv::Mat tximg = transformed->get_image(0);

    cv::Mat output_image(256, 256, CV_32SC1);
    loader.load({output_image.data}, transformed);

    {
        int32_t* output = (int32_t*)(output_image.data);
        int index = 0;
        for(int row = 0; row < 256; row++) {
            for(int col = 0; col < 256; col++) {
                uint8_t value = col;
                ASSERT_EQ(value, output[index++]);
            }
        }
    }
}

//TEST(pixel_mask, read_test) {
//    string path = "/home/users/robert/segnet/";
//    string train = path+"train/";
//    string trainannot = path+"trainannot/";
//    string base_name = "0001TP_006690.png";
//    string train_path = train+base_name;
//    string trainannot_path = trainannot+base_name;

//    nlohmann::json js = {{"width", 512},{"height",256}};
//    image::config cfg(js);

//    auto train_data = read_file_contents(train_path);
//    auto trainannot_data = read_file_contents(trainannot_path);
//    ASSERT_TRUE(train_data.size()>0) << train_path;
//    ASSERT_TRUE(trainannot_data.size()>0) << trainannot_path;

////    cv::Mat mat;
////    cv::imdecode(trainannot_data,CV_LOAD_IMAGE_COLOR,&mat);
////    cout << "pixel size " << mat.elemSize() << endl;

//    image::extractor        extractor{cfg};
//    pixel_mask::transformer transformer{cfg};
//    image::loader           loader{cfg};
//    image::param_factory    factory{cfg};

//    auto extracted = extractor.extract((const char*)trainannot_data.data(), trainannot_data.size());
//    image_params_builder builder(factory.make_params(extracted));
//    shared_ptr<image::params> params_ptr = builder;
//    shared_ptr<image::decoded> transformed = transformer.transform(params_ptr, extracted);
//    cv::Mat tximg = transformed->get_image(0);

////    unsigned char *data = (unsigned char*)(tximg.data);
////    int index = 0;
////    for(int row = 0; row < tximg.rows; row++) {
////        for(int col = 0; col < tximg.cols; col++) {
////            int b = data[index++];
////            int g = data[index++];
////            int r = data[index++];
////            cout << b << "," << g << "," << r << endl;
////        }
////    }
//}
