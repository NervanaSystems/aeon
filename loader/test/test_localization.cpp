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
#include "gen_image.hpp"


#define private public


#include "params.hpp"
#include "etl_image.hpp"
#include "etl_localization.hpp"
#include "json.hpp"

using namespace std;
using namespace nervana;
using namespace nervana::localization;

static vector<string> label_list = {"person","dog","lion","tiger","eel","puma","rat","tick","flea","bicycle","hovercraft"};

static string read_file( const string& path ) {
    ifstream f(path);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static shared_ptr<localization::config> make_localization_config() {
    auto cfg = make_shared<localization::config>();
    nlohmann::json js;
    js["labels"] = label_list;
    cfg->set_config(js);
    return cfg;
}

TEST(localization,generate_anchors) {
    // Verify that we compute the same anchors as Shaoqing's matlab implementation:
    //    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
    //    >> anchors
    //
    //    anchors =
    //       -83   -39   100    56
    //      -175   -87   192   104
    //      -359  -183   376   200
    //       -55   -55    72    72
    //      -119  -119   136   136
    //      -247  -247   264   264
    //       -35   -79    52    96
    //       -79  -167    96   184
    //      -167  -343   184   360
    //
    // base_size 16, ratios [0.5, 1, 2], scales [ 8 16 32]

    vector<box> expected =
                        {{  -83.0-1.0,  -39.0-1.0, 100.0-1.0,  56.0-1.0},
                         { -175.0-1.0,  -87.0-1.0, 192.0-1.0, 104.0-1.0},
                         { -359.0-1.0, -183.0-1.0, 376.0-1.0, 200.0-1.0},
                         {  -55.0-1.0,  -55.0-1.0,  72.0-1.0,  72.0-1.0},
                         { -119.0-1.0, -119.0-1.0, 136.0-1.0, 136.0-1.0},
                         { -247.0-1.0, -247.0-1.0, 264.0-1.0, 264.0-1.0},
                         {  -35.0-1.0,  -79.0-1.0,  52.0-1.0,  96.0-1.0},
                         {  -79.0-1.0, -167.0-1.0,  96.0-1.0, 184.0-1.0},
                         { -167.0-1.0, -343.0-1.0, 184.0-1.0, 360.0-1.0}};

    // subtract 1 from the expected vector as it was generated with 1's based matlab
//    expected -= 1;

    int base_size = 16;
    float scale = 1.0 / 16.;  // scaling factor of the image layers (e.g. VGG)
    vector<float> ratios = {0.5, 1, 2};
    vector<float> scales = {8, 16, 32};

    auto cfg = make_localization_config();

    anchor _anchor{cfg};
    vector<box> actual = _anchor.generate_anchors(base_size, ratios, scales);
    ASSERT_EQ(expected.size(),actual.size());
    for(int i=0; i<expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST(localization,config) {
    nlohmann::json js;
    js["labels"] = label_list;

    cout << js.dump(4) << endl;

    localization::config cfg;
    EXPECT_TRUE(cfg.set_config(js));
}

TEST(localization,calculate_scale_shape) {

    auto cfg = make_localization_config();
    localization::transformer transformer(cfg);
    cv::Size size{500,375};
    float scale;
    tie(scale,size) = transformer.calculate_scale_shape(size);
    EXPECT_FLOAT_EQ(1.6,scale);
    EXPECT_EQ(800,size.width);
    EXPECT_EQ(600,size.height);
}


TEST(localization, transform) {
//    {
//        string data = read_file(CURDIR"/test_data/000001.json");
//        localization::extractor extractor{label_list};
//        localization::transformer transformer;
//        auto mdata = extractor.extract(&data[0],data.size());
//        auto decoded = static_pointer_cast<nervana::localization::decoded>(mdata);
//        ASSERT_NE(nullptr,decoded);
//        auto params = make_shared<image::params>();
//        transformer.transform(params, decoded);
//        auto boxes = decoded->boxes();
//    }
    {
        string data = read_file(CURDIR"/test_data/006637.json");
        auto cfg = make_localization_config();
        localization::extractor extractor{cfg};
        localization::transformer transformer{cfg};
        auto mdata = extractor.extract(&data[0],data.size());
        auto decoded = static_pointer_cast<nervana::localization::decoded>(mdata);
        ASSERT_NE(nullptr,decoded);
        auto params = make_shared<image::params>();
        transformer.transform(params, decoded);
        auto boxes = decoded->boxes();
    }
//    {
//        string data = read_file(CURDIR"/test_data/009952.json");
//        localization::extractor extractor{label_list};
//        auto mdata = extractor.extract(&data[0],data.size());
//        auto decoded = static_pointer_cast<nervana::localization::decoded>(mdata);
//        ASSERT_NE(nullptr,decoded);
//        auto boxes = decoded->boxes();
//    }
}


