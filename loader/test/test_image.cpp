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

class image_config_builder {
public:
    image_config_builder() { height(30); width(30); }

    image_config_builder& height(int val) { obj.height = val; return *this; }
    image_config_builder& width(int val) { obj.width = val; return *this; }
    image_config_builder& do_area_scale(bool val) { obj.do_area_scale = val; return *this; }
    image_config_builder& channel_major(bool val) { obj.channel_major = val; return *this; }
    image_config_builder& channels(int val) { obj.channels = val; return *this; }
    image_config_builder& scale(float a, float b) { obj.scale = uniform_real_distribution<float>(a,b); return *this; }
    image_config_builder& angle(float a, float b) { obj.angle = uniform_int_distribution<int>(a,b); return *this; }
    image_config_builder& lighting(float mean, float stddev) { obj.lighting = normal_distribution<float>(mean,stddev); return *this; }
    image_config_builder& aspect_ratio(float a, float b) { obj.aspect_ratio = uniform_real_distribution<float>(a,b); return *this; }
    image_config_builder& photometric(float a, float b) { obj.photometric = uniform_real_distribution<float>(a,b); return *this; }
    image_config_builder& crop_offset(float a, float b) { obj.crop_offset = uniform_real_distribution<float>(a,b); return *this; }
    image_config_builder& flip(float p) { obj.flip = bernoulli_distribution(p); return *this; }

    shared_ptr<image::config> dump(int tab=4) {
        nlohmann::json js = {{"height",obj.height},{"width",obj.width},{"channels",obj.channels},
        {"do_area_scale",obj.do_area_scale},{"channel_major",obj.channel_major},{
        "distribution",{
            {"angle",{obj.angle.a(),obj.angle.b()}},
            {"scale",{obj.scale.a(),obj.scale.b()}},
            {"lighting",{obj.lighting.mean(),obj.lighting.stddev()}},
            {"aspect_ratio",{obj.aspect_ratio.a(),obj.aspect_ratio.b()}},
            {"photometric",{obj.photometric.a(),obj.photometric.b()}},
            {"crop_offset",{obj.crop_offset.a(),obj.crop_offset.b()}},
            {"flip",{obj.flip.p()!=0}}
        }}};
        auto rc = make_shared<image::config>();
        rc->set_config(js);
        return rc;
    }
private:
    image::config obj;
};

class image_params_builder {
public:
    image_params_builder(shared_ptr<image::params> _obj) { obj = _obj; }

    image_params_builder& cropbox( int x, int y, int w, int h ) { obj->cropbox = cv::Rect(x,y,w,h); return *this; }
    image_params_builder& output_size( int w, int h ) { obj->output_size = cv::Size2i(w,h); return *this; }
    image_params_builder& angle( int val ) { obj->angle = val; return *this; }
    image_params_builder& flip( bool val ) { obj->flip = val; return *this; }
    image_params_builder& lighting( float f1, float f2, float f3 ) { obj->lighting = {f1,f2,f3}; return *this; }
    image_params_builder& color_noise_std(float f) { obj->color_noise_std = f; return *this; }
    image_params_builder& photometric( float f1, float f2, float f3 ) { obj->photometric = {f1,f2,f3}; return *this; }

    operator shared_ptr<image::params>() const {
        return obj;
    }

private:
    shared_ptr<image::params> obj;
};

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

    auto itpj = make_shared<image::config>();
    itpj->set_config(js);

    image::extractor ext{itpj};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    ASSERT_NE(nullptr,decoded);
    EXPECT_EQ(1,decoded->get_image_count());
    cv::Size2i size = decoded->get_image_size();
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

TEST(etl, decoded_image) {
    cv::Mat img1 = cv::Mat( 256, 256, CV_8UC3 );
    cv::Mat img2 = cv::Mat( 256, 256, CV_8UC3 );
    cv::Mat img3 = cv::Mat( 256, 256, CV_8UC3 );
    cv::Mat img4 = cv::Mat( 100, 100, CV_8UC3 );

    vector<cv::Mat> v1{ img1, img2, img3 };
    vector<cv::Mat> v2{ img4 };

    image::decoded decoded;
    EXPECT_TRUE(decoded.add(img1));
    EXPECT_TRUE(decoded.add(img2));
    EXPECT_TRUE(decoded.add(img3));
    EXPECT_TRUE(decoded.add(v1));
    EXPECT_FALSE(decoded.add(img4));    // image size does not match
    EXPECT_FALSE(decoded.add(v2));
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

    auto config = make_shared<image::config>();
    config->set_config(js);
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

bool check_value(shared_ptr<image::decoded> transformed, int x0, int y0, int x1, int y1, int ii=0) {
    cv::Mat image = transformed->get_image(ii);
    cv::Vec3b value = image.at<cv::Vec3b>(y0,x0); // row,col
    return x1 == (int)value[0] && y1 == (int)value[1];
}

TEST(etl, image_transform_crop) {
    auto indexed = generate_indexed_image();
    vector<unsigned char> img;
    cv::imencode( ".png", indexed, img );

    image_config_builder config;
    shared_ptr<image::config> config_ptr = config.width(256).height(256).dump();

    image::extractor ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    image::param_factory factory(config_ptr);

    image_params_builder builder(factory.make_params(decoded));
    shared_ptr<image::params> params_ptr = builder.cropbox( 100, 150, 20, 30 ).output_size(20, 30);

    image::transformer trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(20,image.size().width);
    EXPECT_EQ(30,image.size().height);

    EXPECT_TRUE(check_value(transformed,0,0,100,150));
    EXPECT_TRUE(check_value(transformed,19,0,119,150));
    EXPECT_TRUE(check_value(transformed,0,29,100,179));
}

TEST(etl, image_transform_flip) {
    auto indexed = generate_indexed_image();
    vector<unsigned char> img;
    cv::imencode( ".png", indexed, img );

    image_config_builder config;
    shared_ptr<image::config> config_ptr = config.width(256).height(256).dump();

    image::extractor ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    image::param_factory factory(config_ptr);

    image_params_builder builder(factory.make_params(decoded));
    shared_ptr<image::params> params_ptr = builder.cropbox( 100, 150, 20, 20 ).output_size(20, 20).flip(true);

    image::transformer trans{config_ptr};
    shared_ptr<image::decoded> transformed = trans.transform(params_ptr, decoded);

    cv::Mat image = transformed->get_image(0);
    EXPECT_EQ(20,image.size().width);
    EXPECT_EQ(20,image.size().height);

    EXPECT_TRUE(check_value(transformed,0,0,119,150));
    EXPECT_TRUE(check_value(transformed,19,0,100,150));
    EXPECT_TRUE(check_value(transformed,0,19,119,169));
}

TEST(etl, multi_crop) {
    auto indexed = generate_indexed_image();  // 256 x 256
    vector<unsigned char> img;
    cv::imencode( ".png", indexed, img );

    image_config_builder config;
    shared_ptr<image::config> config_ptr = config.dump();  // Only used for extract and load

    image::extractor ext{config_ptr};
    shared_ptr<image::decoded> decoded = ext.extract((char*)&img[0], img.size());

    // Just center crop
    {
        auto jsstring = R"(
            {
                "width": 224,
                "height": 224,
                "multicrop_scales": [0.875],
                "crops_per_scale": 1
            }
        )";
        auto js = nlohmann::json::parse(jsstring);
        auto mc_config_ptr = make_shared<multicrop::config>();
        mc_config_ptr->set_config(js);

        multicrop::transformer trans{mc_config_ptr};
        shared_ptr<image::decoded> transformed = trans.transform(nullptr, decoded);

        cv::Mat image = transformed->get_image(0);
        EXPECT_EQ(224,image.size().width);
        EXPECT_EQ(224,image.size().height);

        // First image in transformed should be the center crop, unflipped
        EXPECT_TRUE(check_value(transformed,   0,   0,  16,  16));
        EXPECT_TRUE(check_value(transformed, 223, 223, 239, 239));

        // Second image in transformed should be the center crop, flipped
        EXPECT_TRUE(check_value(transformed,   0,   0, 239,  16, 1));
        EXPECT_TRUE(check_value(transformed, 223, 223,  16, 239, 1));

    }

    // Multi crop, no flip
    {
        auto jsstring = R"(
            {
                "width": 224,
                "height": 224,
                "multicrop_scales": [0.875],
                "include_flips": false
            }
        )";

        auto js = nlohmann::json::parse(jsstring);
        auto mc_config_ptr = make_shared<multicrop::config>();
        mc_config_ptr->set_config(js);

        multicrop::transformer trans{mc_config_ptr};
        shared_ptr<image::decoded> transformed = trans.transform(nullptr, decoded);

        cv::Mat image = transformed->get_image(0);
        EXPECT_EQ(224,image.size().width);
        EXPECT_EQ(224,image.size().height);

        EXPECT_EQ(transformed->get_image_count(), 5);
        // First image in transformed should be the center crop, unflipped
        EXPECT_TRUE(check_value(transformed,   0,   0,  16,  16));
        EXPECT_TRUE(check_value(transformed, 223, 223, 239, 239));

        // NW, SW, NE, SE
        EXPECT_TRUE(check_value(transformed,   0,   0,   0,   0, 1));
        EXPECT_TRUE(check_value(transformed,   0,   0,   0,  32, 2));
        EXPECT_TRUE(check_value(transformed,   0,   0,  32,   0, 3));
        EXPECT_TRUE(check_value(transformed,   0,   0,  32,  32, 4));
    }
    // Multi crop, scale
    {
        auto jsstring = R"(
            {
                "width": 112,
                "height": 112,
                "multicrop_scales": [0.875],
                "include_flips": false
            }
        )";
        using namespace cv;
        using idxPt = std::pair<int, Point2i>;

        auto js = nlohmann::json::parse(jsstring);
        auto mc_config_ptr = make_shared<multicrop::config>();
        mc_config_ptr->set_config(js);

        multicrop::transformer trans{mc_config_ptr};
        shared_ptr<image::decoded> transformed = trans.transform(nullptr, decoded);


        EXPECT_EQ(transformed->get_image_count(), 5);

        Mat resize_crop;
        Size2i out_sz(112, 112), crp_sz(224, 224);

        std::vector<idxPt> vip {{0, Point2i(16, 16)},
                                {1, Point2i( 0,  0)},
                                {2, Point2i( 0, 32)},
                                {3, Point2i(32,  0)},
                                {4, Point2i(32, 32)}};

        for (const idxPt &pp: vip) {
            Rect box(pp.second, crp_sz);
            Mat image = transformed->get_image(pp.first);
            cv::resize(indexed(box), resize_crop, out_sz, 0, 0, CV_INTER_AREA);
            EXPECT_EQ(cv::sum(image != resize_crop), Scalar(0,0,0,0));
        }
    }

}
