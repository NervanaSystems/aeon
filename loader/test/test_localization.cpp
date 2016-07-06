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
#include "etl_image_var.hpp"
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

    auto cfg = make_localization_config();

    anchor _anchor{cfg};
    vector<box> actual = _anchor.generate_anchors();
    ASSERT_EQ(expected.size(),actual.size());
    for(int i=0; i<expected.size(); i++) {
        EXPECT_EQ(expected[i], actual[i]);
    }

    EXPECT_EQ(34596,cfg->total_anchors());

    EXPECT_EQ((9 * (62 * 62)),_anchor.all_anchors.size());
}

void plot(const vector<box>& list, const string& prefix) {
    float xmin = 0.0;
    float xmax = 0.0;
    float ymin = 0.0;
    float ymax = 0.0;
    for(const box& b : list) {
        xmin = std::min(xmin,b.xmin);
        xmax = std::max(xmax,b.xmax);
        ymin = std::min(ymin,b.ymin);
        ymax = std::max(ymax,b.ymax);
    }

    cv::Mat img(ymax-ymin, xmax-xmin, CV_8UC3);
    img = cv::Scalar(255,255,255);

    for( box b : list ) {
        b.xmin -= xmin;
        b.xmax -= xmin;
        b.ymin -= ymin;
        b.ymax -= ymin;
        cv::rectangle(img, b.rect(), cv::Scalar(255,0,0));
    }
    box b = list[0];
    b.xmin -= xmin;
    b.xmax -= xmin;
    b.ymin -= ymin;
    b.ymax -= ymin;

    cv::rectangle(img, b.rect(), cv::Scalar(0,0,255));

    string fname = to_string(int(list[0].width())) + "x" + to_string(int(list[0].height())) + ".png";
    fname = prefix + fname;
    cv::imwrite(fname,img);
}

void plot(const string& path) {
    string prefix = path.substr(path.size()-11, 6) + "-";
    string data = read_file(path);
    auto cfg = make_localization_config();
    localization::extractor extractor{cfg};
    localization::transformer transformer{cfg};
    auto extracted_metadata = extractor.extract(&data[0],data.size());
    ASSERT_NE(nullptr,extracted_metadata);
    auto params = make_shared<image_var::params>();
    shared_ptr<localization::decoded> transformed_metadata = transformer.transform(params, extracted_metadata);

    vector<box>& an = extracted_metadata->anchors;

    int last_width = 0;
    int last_height = 0;
    vector<box> list;
    for(const box& b : an) {
        if(last_width != b.width() || last_height != b.height()) {
            if(list.size() > 0) {
                plot(list, prefix);
                list.clear();
            }
        }
        list.push_back(b);
        last_width = b.width();
        last_height = b.height();
    }
    if(list.size() > 0) {
        plot(list, prefix);
    }

    vector<int>    labels       = transformed_metadata->labels;
    vector<target> bbox_targets = transformed_metadata->bbox_targets;
    vector<int>    anchor_index = transformed_metadata->anchor_index;
    vector<box>    all_anchors  = transformed_metadata->anchors;

    an = transformed_metadata->anchors;

//    for(int i=0; i<transformed_metadata->anchor_index.size(); i++) {
//        cout << "loader " << i << " " << transformed_metadata->anchor_index[i] << " " << labels[transformed_metadata->anchor_index[i]] << endl;
//        cout << an[transformed_metadata->anchor_index[i]] << endl;
//    }

    {
        cv::Mat img(extracted_metadata->image_size, CV_8UC3);
        img = cv::Scalar(255,255,255);
        // Draw foreground boxes
        for(int i=0; i<anchor_index.size(); i++) {
            int index = anchor_index[i];
            if(labels[index]==1) {
                box abox = an[index];
                cv::rectangle(img, abox.rect(), cv::Scalar(0,255,0));
            }
        }

        // Draw bounding boxes
        for( box b : extracted_metadata->boxes()) {
            b = b * extracted_metadata->image_scale;
            cv::rectangle(img, b.rect(), cv::Scalar(255,0,0));
        }
        cv::imwrite(prefix+"fg.png",img);
    }

    {
        cv::Mat img(extracted_metadata->image_size, CV_8UC3);
        img = cv::Scalar(255,255,255);
        // Draw background boxes
        for(int i=0; i<anchor_index.size(); i++) {
            int index = anchor_index[i];
            if(labels[index]==0) {
                box abox = an[index];
                cv::rectangle(img, abox.rect(), cv::Scalar(0,0,255));
            }
        }

        // Draw bounding boxes
        for( box b : extracted_metadata->boxes()) {
            b = b * extracted_metadata->image_scale;
            cv::rectangle(img, b.rect(), cv::Scalar(255,0,0));
        }
        cv::imwrite(prefix+"bg.png",img);
    }
}

TEST(DISABLED_localization,plot) {
    plot(CURDIR"/test_data/009952.json");
}

TEST(localization,config) {
    nlohmann::json js;
    js["labels"] = label_list;

    localization::config cfg;
    EXPECT_TRUE(cfg.set_config(js));
}

TEST(localization,calculate_scale_shape) {

    auto cfg = make_localization_config();
    localization::transformer transformer(cfg);
    cv::Size size{500,375};
    float scale;
    tie(scale,size) = transformer.calculate_scale_shape(size, cfg->min_size, cfg->max_size);
    EXPECT_FLOAT_EQ(1.6,scale);
    EXPECT_EQ(800,size.width);
    EXPECT_EQ(600,size.height);
}

TEST(localization, sample_anchors) {
    string data = read_file(CURDIR"/test_data/006637.json");
    shared_ptr<localization::config> cfg = make_localization_config();
    localization::extractor extractor{cfg};
    localization::transformer transformer{cfg};
    auto extracted_metadata = extractor.extract(&data[0],data.size());
    ASSERT_NE(nullptr,extracted_metadata);
    shared_ptr<image_var::params> params = make_shared<image_var::params>();
    auto transformed_metadata = transformer.transform(params, extracted_metadata);
    ASSERT_NE(nullptr,transformed_metadata);

    vector<int>    labels       = transformed_metadata->labels;
    vector<target> bbox_targets = transformed_metadata->bbox_targets;
    vector<int>    anchor_index = transformed_metadata->anchor_index;
    vector<box>    anchors = transformed_metadata->anchors;

    EXPECT_EQ(34596,labels.size());
    EXPECT_EQ(34596,bbox_targets.size());
    EXPECT_EQ(256,anchor_index.size());
    EXPECT_EQ(34596,anchors.size());

    for(int index : anchor_index) {
        EXPECT_GE(index,0);
        EXPECT_LT(index,34596);
    }
    for(int index : anchor_index) {
        box b = anchors[index];
        EXPECT_GE(b.xmin,0);
        EXPECT_GE(b.ymin,0);
        EXPECT_LT(b.xmax,cfg->max_size);
        EXPECT_LT(b.ymax,cfg->max_size);
    }
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
        auto decoded_data = extractor.extract(&data[0],data.size());
        ASSERT_NE(nullptr,decoded_data);
        auto params = make_shared<image_var::params>();
        shared_ptr<localization::decoded> transformed_data = transformer.transform(params, decoded_data);

        vector<int> fg_idx = {
             1200,     1262,     1324,     1386,    23954,    24016,    24078,    24090,
            24140,    24152,    24202,    24214,    24264,    24276,    24338,    24400,
            24462,    24503,    24524,    24565,    24586,    24648,    27977,    27978,
            28039,    28040,    28101,    28102,    28163,    28164,    28225,    28226,
            28287,    28559,    28560
        };

        vector<int> bg_idx = {
            192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
            264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
            280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 316, 317, 318, 319,
            320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
            336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
            352, 353, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
            392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
            408, 409, 410, 411, 412, 413, 414, 415, 440, 441, 442, 443, 444, 445, 446, 447,
            448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463,
            464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 502, 503,
            504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
            520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532
        };

        ASSERT_EQ(transformed_data->anchor_index.size(), fg_idx.size() + bg_idx.size());


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

TEST(localization, loader) {
    string data = read_file(CURDIR"/test_data/006637.json");
    auto cfg = make_localization_config();
    localization::extractor extractor{cfg};
    localization::transformer transformer{cfg};
    localization::loader loader{cfg};
    auto mdata = extractor.extract(&data[0],data.size());
    auto decoded = static_pointer_cast<nervana::localization::decoded>(mdata);
    ASSERT_NE(nullptr,decoded);
    auto params = make_shared<image_var::params>();
    shared_ptr<localization::decoded> transformed = transformer.transform(params, decoded);
    loader.load(nullptr, transformed);
}

TEST(localization, compute_targets) {
    // expected values generated via python localization example

    vector<box> gt_bb;
    vector<box> rp_bb;

    // ('gt_bb {0}', array([ 561.6,  329.6,  713.6,  593.6]))
    // ('rp_bb {1}', array([ 624.,  248.,  799.,  599.]))
    // xgt 638.1, rp 712.0, dx -0.419886363636
    // ygt 462.1, rp 424.0, dy  0.108238636364
    // wgt 153.0, rp 176.0, dw -0.140046073646
    // hgt 265.0, rp 352.0, dh -0.283901349612

    gt_bb.emplace_back(561.6,  329.6,  713.6,  593.6);
    rp_bb.emplace_back(624.,  248.,  799.,  599.);

    float dx_0_expected = -0.419886363636;
    float dy_0_expected =  0.108238636364;
    float dw_0_expected = -0.140046073646;
    float dh_0_expected = -0.283901349612;

    // ('gt_bb {0}', array([ 561.6,  329.6,  713.6,  593.6]))
    // ('rp_bb {1}', array([ 496.,  248.,  671.,  599.]))
    // xgt 638.1, rp 584.0, dx  0.307386363636
    // ygt 462.1, rp 424.0, dy  0.108238636364
    // wgt 153.0, rp 176.0, dw -0.140046073646
    // hgt 265.0, rp 352.0, dh -0.283901349612

    gt_bb.emplace_back(561.6,  329.6,  713.6,  593.6);
    rp_bb.emplace_back(496.,  248.,  671.,  599.);

    float dx_1_expected =  0.307386363636;
    float dy_1_expected =  0.108238636364;
    float dw_1_expected = -0.140046073646;
    float dh_1_expected = -0.283901349612;

    ASSERT_EQ(gt_bb.size(), rp_bb.size());

    vector<target> result = localization::transformer::compute_targets(gt_bb, rp_bb);
    ASSERT_EQ(result.size(), gt_bb.size());

    float acceptable_error = 0.00001;

    EXPECT_NEAR(dx_0_expected, result[0].dx, acceptable_error);
    EXPECT_NEAR(dy_0_expected, result[0].dy, acceptable_error);
    EXPECT_NEAR(dw_0_expected, result[0].dw, acceptable_error);
    EXPECT_NEAR(dh_0_expected, result[0].dh, acceptable_error);
    EXPECT_NEAR(dx_1_expected, result[1].dx, acceptable_error);
    EXPECT_NEAR(dy_1_expected, result[1].dy, acceptable_error);
    EXPECT_NEAR(dw_1_expected, result[1].dw, acceptable_error);
    EXPECT_NEAR(dh_1_expected, result[1].dh, acceptable_error);
}

