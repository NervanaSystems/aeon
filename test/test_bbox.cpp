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
#include <random>
#include <algorithm>

#include "gtest/gtest.h"
#include "gen_image.hpp"
#include "cpio.hpp"

#define private public
#define protected public

#include "file_util.hpp"
#include "interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_boundingbox.hpp"
#include "etl_label_map.hpp"
#include "json.hpp"
#include "helpers.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

using bbox = boundingbox::box;
using nbox = normalized_box::box;

static vector<string> label_list = {"person",
                                    "dog",
                                    "lion",
                                    "tiger",
                                    "eel",
                                    "puma",
                                    "rat",
                                    "tick",
                                    "flea",
                                    "bicycle",
                                    "hovercraft"};

static boundingbox::config make_bbox_config(int max_boxes)
{
    nlohmann::json obj = {{"height", 100}, {"width", 150}, {"max_bbox_count", max_boxes}};
    obj["class_names"] = label_list;
    return boundingbox::config(obj);
}

cv::Mat draw(int width, int height, const vector<bbox>& blist, cv::Rect crop = cv::Rect())
{
    cv::Mat image = cv::Mat(width, height, CV_8UC3);
    image         = cv::Scalar(255, 255, 255);
    for (auto box : blist)
    {
        cv::rectangle(image, box.rect(), cv::Scalar(255, 0, 0));
    }
    if (crop != cv::Rect())
    {
        cv::rectangle(image, crop, cv::Scalar(0, 0, 255));
    }
    return image;
}

shared_ptr<augment::image::params> make_params(int width, int height)
{
    shared_ptr<augment::image::params> iparam = make_shared<augment::image::params>();
    iparam->cropbox                           = cv::Rect(0, 0, width, height);
    iparam->output_size                       = cv::Size(width, height);
    return iparam;
}

TEST(boundingbox, extractor)
{
    {
        string data = file_util::read_file_to_string(CURDIR "/test_data/000001.json");
        auto   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(2, boxes.size());

        EXPECT_EQ(194, boxes[0].xmax());
        EXPECT_EQ(47, boxes[0].xmin());
        EXPECT_EQ(370, boxes[0].ymax());
        EXPECT_EQ(239, boxes[0].ymin());
        EXPECT_FALSE(boxes[0].difficult());
        EXPECT_TRUE(boxes[0].truncated());
        EXPECT_EQ(1, boxes[0].label());

        EXPECT_EQ(351, boxes[1].xmax());
        EXPECT_EQ(7, boxes[1].xmin());
        EXPECT_EQ(497, boxes[1].ymax());
        EXPECT_EQ(11, boxes[1].ymin());
        EXPECT_FALSE(boxes[1].difficult());
        EXPECT_TRUE(boxes[1].truncated());
        EXPECT_EQ(0, boxes[1].label());
    }
    {
        string data = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
        auto   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(6, boxes.size());

        auto b = boxes[3];
        EXPECT_EQ(365, b.xmax());
        EXPECT_EQ(324, b.xmin());
        EXPECT_EQ(315, b.ymax());
        EXPECT_EQ(109, b.ymin());
        EXPECT_FALSE(b.difficult());
        EXPECT_FALSE(b.truncated());
        EXPECT_EQ(0, b.label());

        auto c = b * 2.0f;
        EXPECT_EQ(b.xmax() * 2.0f, c.xmax());

        cv::Size2i ref(500, 375);
        EXPECT_EQ(ref, decoded->image_size());

        EXPECT_EQ(3, decoded->depth());
        std::stringstream ss;
        ss << b;
        EXPECT_EQ(ss.str(), "[42 x 207 from (324, 109)] label=0 difficult=0 truncated=0");
    }
    {
        string data = file_util::read_file_to_string(CURDIR "/test_data/009952.json");
        auto   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(1, boxes.size());
    }
}

TEST(boundingbox, meet_emit_constraint_center)
{
    nlohmann::json aug = {
        {"type", "image"}, {"emit_constraint_type", "center"}, {"expand_probability", 0}};
    augment::image::param_factory      factory(aug);
    shared_ptr<augment::image::params> iparam =
        factory.make_ssd_params(256, 256, 8, 8, vector<bbox>());
    iparam->cropbox = cv::Rect(6, 6, 8, 8);

    // Create test metadata
    cv::Rect r0   = cv::Rect(0, 0, 10, 10);
    cv::Rect r1   = cv::Rect(10, 10, 20, 20);
    cv::Rect r2   = cv::Rect(5, 5, 10, 10);
    cv::Rect r3   = cv::Rect(9, 9, 2, 2);
    auto     list = {create_box(r0, "rat"),
                 create_box(r1, "flea"),
                 create_box(r2, "tick"),
                 create_box(r3, "lion")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    boundingbox::transformer transform(cfg);

    auto         tx_decoded = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes   = tx_decoded->boxes();

    EXPECT_EQ(tx_boxes.size(), 2);
    EXPECT_EQ(tx_boxes[0].rect(), cv::Rect(0, 0, 8, 8));
    EXPECT_EQ(tx_boxes[1].rect(), cv::Rect(3, 3, 2, 2));
}

TEST(boundingbox, meet_emit_constraint_min_overlap)
{
    nlohmann::json aug = {{"type", "image"},
                          {"expand_probability", 0},
                          {"emit_constraint_type", "MIN_OVERLAP"},
                          {"emit_constraint_min_overlap", 0.9}};
    augment::image::param_factory      factory(aug);
    shared_ptr<augment::image::params> iparam =
        factory.make_ssd_params(256, 256, 10, 1, vector<bbox>());
    iparam->cropbox = cv::Rect(0, 1, 10, 1);

    // Create test metadata
    cv::Rect r0   = cv::Rect(0, 1, 10, 1); // 10/10
    cv::Rect r1   = cv::Rect(0, 0, 10, 1); // 0/20
    cv::Rect r2   = cv::Rect(0, 2, 8, 1);  // 8/10
    cv::Rect r3   = cv::Rect(0, 1, 12, 1); // 10/12
    cv::Rect r4   = cv::Rect(0, 1, 11, 1); // 10/11
    auto     list = {create_box(r0, "rat"),
                 create_box(r1, "flea"),
                 create_box(r2, "tick"),
                 create_box(r3, "lion"),
                 create_box(r4, "person")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    boundingbox::transformer transform(cfg);

    auto         tx_decoded = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes   = tx_decoded->boxes();

    ASSERT_EQ(tx_boxes.size(), 2);
    EXPECT_EQ(tx_boxes[0].rect(), cv::Rect(0, 0, 10, 1));
    EXPECT_EQ(tx_boxes[1].rect(), cv::Rect(0, 0, 10, 1));
}

TEST(boundingbox, operator_mult)
{
    nervana::box a(1.0, 2.0, 3.0, 4.0);
    auto         b = a * 4.0;
    EXPECT_EQ(4, b.xmin());
    EXPECT_EQ(8, b.ymin());
}

TEST(boundingbox, extractor_error)
{
    string                 data = "{}";
    auto                   cfg  = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    EXPECT_THROW(extractor.extract(&data[0], data.size()), std::invalid_argument);
}

TEST(boundingbox, extractor_missing_label)
{
    string         data        = file_util::read_file_to_string(CURDIR "/test_data/009952.json");
    nlohmann::json obj         = {{"height", 100}, {"width", 150}, {"max_bbox_count", 20}};
    obj["class_names"]         = {"monkey", "tuna"};
    auto                   cfg = boundingbox::config(obj);
    boundingbox::extractor extractor{cfg.label_map};
    EXPECT_THROW(extractor.extract(&data[0], data.size()), std::invalid_argument);
}

TEST(boundingbox, bbox)
{
    // Create test metadata
    cv::Rect r0   = cv::Rect(0, 0, 10, 15);
    cv::Rect r1   = cv::Rect(10, 10, 12, 13);
    cv::Rect r2   = cv::Rect(100, 100, 120, 130);
    auto     list = {create_box(r0, "rat"), create_box(r1, "flea"), create_box(r2, "tick")};
    auto     j    = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();
    ASSERT_EQ(3, boxes.size());
    EXPECT_EQ(r0, boxes[0].rect());
    EXPECT_EQ(r1, boxes[1].rect());
    EXPECT_EQ(r2, boxes[2].rect());
    EXPECT_EQ(6, boxes[0].label());
    EXPECT_EQ(8, boxes[1].label());
    EXPECT_EQ(7, boxes[2].label());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    auto                               tx     = transform.transform(iparam, decoded);
}

TEST(boundingbox, crop)
{
    // Create test metadata
    cv::Rect r0 = cv::Rect(10, 10, 10, 10); // outside
    cv::Rect r1 = cv::Rect(30, 30, 10, 10); // result[0]

    cv::Rect r2 = cv::Rect(50, 50, 10, 10); // result[1]
    cv::Rect r3 = cv::Rect(69, 31, 10, 10); // result[2]
    cv::Rect r4 = cv::Rect(69, 69, 10, 10); // result[3]

    cv::Rect r5   = cv::Rect(90, 35, 10, 10); // outside
    cv::Rect r6   = cv::Rect(30, 70, 10, 10); // result[4]
    cv::Rect r7   = cv::Rect(30, 30, 40, 40); // result[5]
    auto     list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->cropbox                           = cv::Rect(35, 35, 40, 40);

    auto d = draw(256, 256, decoded->boxes(), iparam->cropbox);
    cv::imwrite("bbox_crop.png", d);

    iparam->output_size     = iparam->cropbox.size();
    auto         tx_decoded = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes   = tx_decoded->boxes();
    ASSERT_EQ(6, tx_boxes.size());

    EXPECT_EQ(cv::Rect(0, 0, 5, 5), tx_boxes[0].rect()) << "1";
    EXPECT_EQ(cv::Rect(15, 15, 10, 10), tx_boxes[1].rect()) << "2";
    EXPECT_EQ(cv::Rect(34, 0, 6, 6), tx_boxes[2].rect()) << "3";
    EXPECT_EQ(cv::Rect(34, 34, 6, 6), tx_boxes[3].rect()) << "4";
    EXPECT_EQ(cv::Rect(0, 35, 5, 5), tx_boxes[4].rect()) << "5";
    EXPECT_EQ(cv::Rect(0, 0, 35, 35), tx_boxes[5].rect()) << "6";
}

TEST(boundingbox, expand)
{
    int   width        = 100;
    int   height       = 100;
    int   out_width    = 200;
    int   out_height   = 200;
    float expand_ratio = static_cast<float>(out_width) / width;
    int   expand_off   = 50;
    // Create test metadata
    vector<cv::Point2f> b_off = {cv::Point2f(10., 10.),
                                 cv::Point2f(30., 30.),
                                 cv::Point2f(50., 50.),
                                 cv::Point2f(70., 30.),
                                 cv::Point2f(90., 35.),
                                 cv::Point2f(30., 70.),
                                 cv::Point2f(70., 70.),
                                 cv::Point2f(30., 30.)};
    vector<cv::Point2f> b_size = {cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(70., 70.)};
    vector<string>         b_label = {"lion", "tiger", "eel", "eel", "eel", "eel", "eel", "eel"};
    vector<nlohmann::json> list;
    vector<bbox>           exp_b;
    for (int i = 0; i < b_off.size(); i++)
    {
        list.push_back(create_box(bbox(b_off[i].x,
                                       b_off[i].y,
                                       b_off[i].x + b_size[i].x - 1,
                                       b_off[i].y + b_size[i].y - 1,
                                       false),
                                  b_label[i]));
        exp_b.push_back(bbox((b_off[i].x + expand_off),
                             (b_off[i].y + expand_off),
                             (b_off[i].x + b_size[i].x - 1 + expand_off),
                             (b_off[i].y + b_size[i].y - 1 + expand_off),
                             false));
    }

    auto j = create_metadata(list, width, height);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(b_off.size(), boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(width, height);
    iparam->output_size                       = cv::Size(out_width, out_height);
    iparam->expand_ratio                      = expand_ratio;
    iparam->expand_offset                     = cv::Size(expand_off, expand_off);
    iparam->expand_size                       = cv::Size(out_width, out_height);
    iparam->cropbox                           = cv::Rect(0, 0, out_width, out_height);

    auto         tx_decoded = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes   = tx_decoded->boxes();

    ASSERT_EQ(b_off.size(), tx_boxes.size());

    for (int i = 0; i < exp_b.size(); i++)
    {
        stringstream ss;
        ss << " at iteration " << i;
        EXPECT_EQ(exp_b[i].rect(), tx_boxes[i].rect()) << ss.str();
    }
}

TEST(boundingbox, rescale)
{
    // Create test metadata
    cv::Rect r0   = cv::Rect(10, 10, 10, 10); // result[0]
    cv::Rect r1   = cv::Rect(30, 30, 10, 10); // result[1]
    cv::Rect r2   = cv::Rect(50, 50, 10, 10); // result[2]
    cv::Rect r3   = cv::Rect(70, 30, 10, 10); // result[3]
    cv::Rect r4   = cv::Rect(90, 35, 10, 10); // result[4]
    cv::Rect r5   = cv::Rect(30, 70, 10, 10); // result[5]
    cv::Rect r6   = cv::Rect(70, 70, 10, 10); // result[6]
    cv::Rect r7   = cv::Rect(30, 30, 80, 80); // result[7]
    auto     list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->output_size                       = cv::Size(512, 1024);
    auto         tx_decoded                   = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes                     = tx_decoded->boxes();
    float        xscale                       = 512. / 256.;
    float        yscale                       = 1024. / 256.;
    ASSERT_EQ(8, tx_boxes.size());
    EXPECT_EQ(cv::Rect(10 * xscale, 10 * yscale, 10 * xscale, 10 * yscale), tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect(30 * xscale, 30 * yscale, 10 * xscale, 10 * yscale), tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect(50 * xscale, 50 * yscale, 10 * xscale, 10 * yscale), tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect(70 * xscale, 30 * yscale, 10 * xscale, 10 * yscale), tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect(90 * xscale, 35 * yscale, 10 * xscale, 10 * yscale), tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect(30 * xscale, 70 * yscale, 10 * xscale, 10 * yscale), tx_boxes[5].rect());
    EXPECT_EQ(cv::Rect(70 * xscale, 70 * yscale, 10 * xscale, 10 * yscale), tx_boxes[6].rect());
    EXPECT_EQ(cv::Rect(30 * xscale, 30 * yscale, 80 * xscale, 80 * yscale), tx_boxes[7].rect());
}

TEST(boundingbox, flip)
{
    // Create test metadata
    cv::Rect r0   = cv::Rect(10, 10, 10, 10); // outside
    cv::Rect r1   = cv::Rect(30, 30, 10, 10); // result[0]
    cv::Rect r2   = cv::Rect(50, 50, 10, 10); // result[1]
    cv::Rect r3   = cv::Rect(70, 30, 10, 10); // result[2]
    cv::Rect r4   = cv::Rect(90, 35, 10, 10); // outside
    cv::Rect r5   = cv::Rect(30, 70, 10, 10); // result[3]
    cv::Rect r6   = cv::Rect(70, 70, 10, 10); // result[4]
    cv::Rect r7   = cv::Rect(30, 30, 80, 80); // result[5]
    auto     list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->flip                              = 1;
    auto         tx_decoded                   = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes                     = tx_decoded->boxes();
    ASSERT_EQ(8, tx_boxes.size());

    EXPECT_EQ(cv::Rect(256 - 10 - 10, 10, 10, 10), tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 30, 30, 10, 10), tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 50, 50, 10, 10), tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 70, 30, 10, 10), tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 90, 35, 10, 10), tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 30, 70, 10, 10), tx_boxes[5].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 70, 70, 10, 10), tx_boxes[6].rect());
    EXPECT_EQ(cv::Rect(256 - 80 - 30, 30, 80, 80), tx_boxes[7].rect());
}

TEST(boundingbox, crop_flip)
{
    // Create test metadata
    cv::Rect r0 = cv::Rect(10, 10, 10, 10); // outside
    cv::Rect r1 = cv::Rect(30, 30, 10, 10); // result[0]

    cv::Rect r2 = cv::Rect(50, 50, 10, 10); // result[1]
    cv::Rect r3 = cv::Rect(69, 31, 10, 10); // result[2]
    cv::Rect r4 = cv::Rect(69, 69, 10, 10); // result[3]

    cv::Rect r5 = cv::Rect(90, 35, 10, 10); // outside
    cv::Rect r6 = cv::Rect(30, 70, 10, 10); // result[4]
    cv::Rect r7 = cv::Rect(30, 30, 40, 40); // result[5]

    auto list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto j = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->cropbox                           = cv::Rect(35, 35, 40, 40);
    iparam->output_size                       = cv::Size(40, 40);
    iparam->flip                              = 1;
    auto         tx_decoded                   = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes                     = tx_decoded->boxes();
    ASSERT_EQ(6, tx_boxes.size());

    EXPECT_EQ(cv::Rect(35, 0, 5, 5), tx_boxes[0].rect()) << "1";
    EXPECT_EQ(cv::Rect(15, 15, 10, 10), tx_boxes[1].rect()) << "2";
    EXPECT_EQ(cv::Rect(00, 00, 06, 06), tx_boxes[2].rect()) << "3";
    EXPECT_EQ(cv::Rect(00, 34, 06, 06), tx_boxes[3].rect()) << "4";
    EXPECT_EQ(cv::Rect(35, 35, 5, 5), tx_boxes[4].rect()) << "5";
    EXPECT_EQ(cv::Rect(05, 00, 35, 35), tx_boxes[5].rect()) << "6";
}

TEST(boundingbox, expand_crop_flip_resize)
{
    int       input_width   = 100;
    int       input_height  = 100;
    int       output_width  = 200;
    int       output_height = 200;
    int       expand_size   = 300;
    float     expand_ratio  = expand_size / input_width;
    cv::Point expand_offset(100, 100);
    cv::Rect  cropbox(50, 50, 150, 150);
    float     xscale = static_cast<float>(output_width) / cropbox.width;
    float     yscale = static_cast<float>(output_height) / cropbox.height;
    // Create test metadata
    vector<cv::Point2f> b_pos = {cv::Point2f(0, 0),
                                 cv::Point2f(10., 10.),
                                 cv::Point2f(30., 30.),
                                 cv::Point2f(50., 50.),
                                 cv::Point2f(70., 30.),
                                 cv::Point2f(30., 70.),
                                 cv::Point2f(70., 70.)};
    vector<cv::Point2f> b_size = {cv::Point2f(10, 10),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(10., 10.),
                                  cv::Point2f(20., 20.)};
    vector<string>         b_label = {"lion", "tiger", "eel", "eel", "eel", "eel", "eel", "eel"};
    vector<nlohmann::json> input_b;
    vector<bbox>           expect_b;
    for (int i = 0; i < b_pos.size(); i++)
    {
        input_b.push_back(create_box(bbox(b_pos[i].x,
                                          b_pos[i].y,
                                          (b_pos[i].x + b_size[i].x - 1),
                                          (b_pos[i].y + b_size[i].y - 1),
                                          false),
                                     b_label[i]));
        // expand and crop
        expect_b.push_back(bbox((b_pos[i].x + expand_offset.x - cropbox.x),
                                (b_pos[i].y + expand_offset.y - cropbox.y),
                                (b_pos[i].x + expand_offset.x - cropbox.x + b_size[i].x - 1),
                                (b_pos[i].y + expand_offset.y - cropbox.y + b_size[i].y - 1),
                                false));

        // flip and resize
        bbox eb(expect_b.back());
        expect_b.back() = bbox((cropbox.width - eb.xmax() - 1) * xscale,
                               eb.ymin() * yscale,
                               ((cropbox.width - eb.xmin() - 1) + 1) * xscale - 1,
                               (eb.ymax() + 1) * yscale - 1,
                               false);
    }

    auto j = create_metadata(input_b, input_width, input_height);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_shared<augment::image::params>();
    iparam->flip                              = true;
    iparam->output_size                       = cv::Size(output_width, output_height);
    iparam->expand_ratio                      = expand_ratio;
    iparam->expand_offset                     = cv::Size(expand_offset.x, expand_offset.y);
    iparam->expand_size                       = cv::Size(expand_size, expand_size);
    iparam->cropbox                           = cropbox;

    auto         tx_decoded = transform.transform(iparam, decoded);
    vector<bbox> tx_boxes   = tx_decoded->boxes();

    ASSERT_EQ(expect_b.size(), tx_boxes.size());

    for (int i = 0; i < expect_b.size(); i++)
    {
        stringstream ss;
        ss << " at iteration " << i;
        EXPECT_EQ(expect_b[i].rect(), tx_boxes[i].rect()) << ss.str();
    }
}

TEST(boundingbox, angle)
{
    // Create test metadata
    cv::Rect r0   = cv::Rect(10, 10, 10, 10);
    auto     list = {create_box(r0, "puma")};
    auto     j    = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                   cfg = make_bbox_config(100);
    boundingbox::extractor extractor{cfg.label_map};
    auto                   decoded = extractor.extract(&buffer[0], buffer.size());
    vector<bbox>           boxes   = decoded->boxes();

    ASSERT_EQ(1, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->angle                             = 5;
    auto tx_decoded                           = transform.transform(iparam, decoded);
    EXPECT_EQ(nullptr, tx_decoded.get());
}

void test_values(const cv::Rect& r, float* outbuf)
{
    EXPECT_EQ(r.x, static_cast<int>(outbuf[0]));
    EXPECT_EQ(r.y, static_cast<int>(outbuf[1]));
    EXPECT_EQ(r.x + r.width - 1, static_cast<int>(outbuf[2]));
    EXPECT_EQ(r.y + r.height - 1, static_cast<int>(outbuf[3]));
}

TEST(boundingbox, load_pad)
{
    cv::Rect r0   = cv::Rect(10, 10, 10, 10); // outside
    cv::Rect r1   = cv::Rect(30, 30, 10, 10); // result[0]
    cv::Rect r2   = cv::Rect(50, 50, 10, 10); // result[1]
    cv::Rect r3   = cv::Rect(70, 30, 10, 10); // result[2]
    cv::Rect r4   = cv::Rect(90, 35, 10, 10); // outside
    cv::Rect r5   = cv::Rect(30, 70, 10, 10); // result[3]
    cv::Rect r6   = cv::Rect(70, 70, 10, 10); // result[4]
    cv::Rect r7   = cv::Rect(30, 30, 80, 80); // result[5]
    auto     list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto   j      = create_metadata(list, 256, 256);
    string buffer = j.dump();

    size_t bbox_max = 10;
    auto   cfg      = make_bbox_config(bbox_max);

    boundingbox::extractor extractor{cfg.label_map};
    boundingbox::loader    loader{cfg};

    vector<float> outbuf(bbox_max * 4 + 1); // xmin, ymin, xmax, ymax
    outbuf[outbuf.size() - 1] = -1;         // one past the end of the buffer
    auto extracted            = extractor.extract(buffer.data(), buffer.size());
    loader.load({reinterpret_cast<char*>(outbuf.data())}, extracted);

    test_values(r0, &outbuf[0]);
    test_values(r1, &outbuf[4]);
    test_values(r2, &outbuf[8]);
    test_values(r3, &outbuf[12]);
    test_values(r4, &outbuf[16]);
    test_values(r5, &outbuf[20]);
    test_values(r6, &outbuf[24]);
    test_values(r7, &outbuf[28]);

    for (int i = 32; i < 40; i++)
    {
        EXPECT_EQ(0, static_cast<int>(outbuf[i]));
    }

    EXPECT_EQ(-1, static_cast<int>(outbuf[outbuf.size() - 1]));
}

TEST(boundingbox, load_full)
{
    cv::Rect r0   = cv::Rect(10, 10, 10, 10); // outside
    cv::Rect r1   = cv::Rect(30, 30, 10, 10); // result[0]
    cv::Rect r2   = cv::Rect(50, 50, 10, 10); // result[1]
    cv::Rect r3   = cv::Rect(70, 30, 10, 10); // result[2]
    cv::Rect r4   = cv::Rect(90, 35, 10, 10); // outside
    cv::Rect r5   = cv::Rect(30, 70, 10, 10); // result[3]
    cv::Rect r6   = cv::Rect(70, 70, 10, 10); // result[4]
    cv::Rect r7   = cv::Rect(30, 30, 80, 80); // result[5]
    auto     list = {create_box(r0, "lion"),
                 create_box(r1, "tiger"),
                 create_box(r2, "eel"),
                 create_box(r3, "eel"),
                 create_box(r4, "eel"),
                 create_box(r5, "eel"),
                 create_box(r6, "eel"),
                 create_box(r7, "eel")};
    auto   j      = create_metadata(list, 256, 256);
    string buffer = j.dump();

    size_t bbox_max = 6;
    auto   cfg      = make_bbox_config(bbox_max);

    boundingbox::extractor extractor{cfg.label_map};
    boundingbox::loader    loader{cfg};

    vector<float> outbuf(bbox_max * 4 + 1); // xmin, ymin, xmax, ymax
    outbuf[outbuf.size() - 1] = -1;         // one past the end of the buffer
    auto extracted            = extractor.extract(buffer.data(), buffer.size());
    loader.load({reinterpret_cast<char*>(outbuf.data())}, extracted);

    test_values(r0, &outbuf[0]);
    test_values(r1, &outbuf[4]);
    test_values(r2, &outbuf[8]);
    test_values(r3, &outbuf[12]);
    test_values(r4, &outbuf[16]);
    test_values(r5, &outbuf[20]);

    EXPECT_EQ(-1, static_cast<int>(outbuf[outbuf.size() - 1]));
}

TEST(boundingbox, intersect_boundingbox)
{
    std::vector<std::tuple<bbox, bbox, bbox>> boxes = {
        std::make_tuple(bbox(5, 5, 60, 60), bbox(50, 40, 80, 70), bbox(50, 40, 60, 60)),
        std::make_tuple(bbox(), bbox(), bbox()),
        std::make_tuple(bbox(0, 0, 0, 0), bbox(0, 0, 0, 0), bbox(0, 0, 0, 0)),
        std::make_tuple(
            bbox(0.7, 0.7, 0.7, 0.7), bbox(0.7, 0.7, 0.7, 0.7), bbox(0.7, 0.7, 0.7, 0.7))};

    for (int i = 0; i < boxes.size(); i++)
    {
        stringstream ss;
        ss << " at iteration " << i;
        box exp_b = get<2>(boxes[i]);
        box int_b = get<0>(boxes[i]).intersect(get<1>(boxes[i]));
        EXPECT_FLOAT_EQ(exp_b.xmin(), int_b.xmin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.xmax(), int_b.xmax()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymin(), int_b.ymin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymax(), int_b.ymax()) << ss.str();
        int_b = get<1>(boxes[i]).intersect(get<0>(boxes[i]));
        EXPECT_FLOAT_EQ(exp_b.xmin(), int_b.xmin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.xmax(), int_b.xmax()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymin(), int_b.ymin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymax(), int_b.ymax()) << ss.str();
    }
}

TEST(normalized_gbox, intersect_normalized_box)
{
    nbox box1(0.0, 0.0, 0.1, 0.1);
    std::vector<std::tuple<nbox, nbox, nbox>> boxes = {
        std::make_tuple(box1, nbox(0.0, 0.0, .05, 0.1), nbox(0.0, 0.0, .05, 0.1)),
        std::make_tuple(box1, nbox(0.0, 0.0, .05, .05), nbox(0.0, 0.0, .05, .05)),
        std::make_tuple(box1, nbox(), nbox()),
        std::make_tuple(box1, nbox(0.2, 0.2, 0.3, 0.3), nbox()),
        std::make_tuple(box1, nbox(0.0, 0.0, 0.1, 0.1), nbox(0.0, 0.0, 0.1, 0.1)),
        std::make_tuple(nbox(0, 0, .6, .6), nbox(.5, .5, 1., 1.), nbox(.5, .5, .6, .6)),
        std::make_tuple(nbox(0.5, 0.5, 1, 1), nbox(0, 0, 0.4, 0.4), nbox())};

    for (int i = 0; i < boxes.size(); i++)
    {
        stringstream ss;
        ss << " at iteration " << i;
        nbox exp_b = get<2>(boxes[i]);
        nbox int_b = get<0>(boxes[i]).intersect(get<1>(boxes[i]));
        EXPECT_FLOAT_EQ(exp_b.xmin(), int_b.xmin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.xmax(), int_b.xmax()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymin(), int_b.ymin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymax(), int_b.ymax()) << ss.str();
        int_b = get<1>(boxes[i]).intersect(get<0>(boxes[i]));
        EXPECT_FLOAT_EQ(exp_b.xmin(), int_b.xmin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.xmax(), int_b.xmax()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymin(), int_b.ymin()) << ss.str();
        EXPECT_FLOAT_EQ(exp_b.ymax(), int_b.ymax()) << ss.str();
    }
}

TEST(boundingbox, jaccard_overlap_boundingbox)
{
    std::vector<std::tuple<bbox, bbox, float>> boxes = {
        std::make_tuple(
            bbox(0, 0, 60, 60), bbox(50, 50, 100, 100), 11.f * 11 / (61 * 61 + 51 * 51 - 11 * 11)),
        std::make_tuple(bbox(), bbox(), 0),
        std::make_tuple(bbox(0, 0, 0, 0), bbox(0, 0, 0, 0), 1)};
    for (int i = 0; i < boxes.size(); i++)
    {
        bbox  b1      = get<0>(boxes[i]);
        bbox  b2      = get<1>(boxes[i]);
        float overlap = get<2>(boxes[i]);
        EXPECT_FLOAT_EQ(b1.jaccard_overlap(b2), overlap) << "at index " << i;
        EXPECT_FLOAT_EQ(b2.jaccard_overlap(b1), overlap) << "at index " << i;
    }
}

TEST(normalized_box, jaccard_overlap_normalized_box)
{
    nbox box1(0, 0, 0.1, 0.1);
    std::vector<std::tuple<nbox, nbox, float>> boxes = {
        std::make_tuple(box1, nbox(0.0, 0.0, .05, 0.1), 0.5),
        std::make_tuple(box1, box1, 1.0),
        std::make_tuple(box1, nbox(0.0, 0.0, .05, .05), .25),
        std::make_tuple(box1, nbox(), 0.0),
        std::make_tuple(box1, nbox(0.2, 0.2, 0.3, 0.3), 0.0),
        std::make_tuple(nbox(0, 0, 0.6, 0.6),
                        nbox(0.5, 0.5, 1, 1),
                        0.1 * 0.1 / (0.5 * 0.5 + 0.6 * 0.6 - 0.1 * 0.1))};
    for (int i = 0; i < boxes.size(); i++)
    {
        nbox  b1      = get<0>(boxes[i]);
        nbox  b2      = get<1>(boxes[i]);
        float overlap = get<2>(boxes[i]);
        EXPECT_FLOAT_EQ(b1.jaccard_overlap(b2), overlap) << "at index " << i;
        EXPECT_FLOAT_EQ(b2.jaccard_overlap(b1), overlap) << "at index " << i;
    }
}
TEST(normalized_box, coverage_normalized_box)
{
    nbox box1(0, 0, 1, 1);
    nbox box2(0, 0, 1, 1);
    nbox box3(0, 0, 0.5, 1);
    nbox box4(0, 0, 0.5, 0.5);
    nbox box5(0.5, 0.5, 1, 1);
    nbox box6(0, 0, 0.6, 0.6);
    std::vector<std::tuple<nbox, nbox, float>> boxes = {
        make_tuple(box1, box2, 1),
        make_tuple(box2, box1, 1),
        make_tuple(box1, box3, 0.5),
        make_tuple(box3, box1, 1),
        make_tuple(box4, box5, 0),
        make_tuple(box5, box4, 0),
        make_tuple(box5, box6, (0.1f * 0.1f) / (0.5f * 0.5f)),
        make_tuple(box6, box5, (0.1f * 0.1f) / (0.6f * 0.6f)),
        make_tuple(nbox(), nbox(), 0)};

    for (int i = 0; i < boxes.size(); i++)
    {
        nbox  b1    = get<0>(boxes[i]);
        nbox  b2    = get<1>(boxes[i]);
        float cover = get<2>(boxes[i]);
        EXPECT_NEAR(b1.coverage(b2), cover, nervana::epsilon) << "at index " << i;
    }
}

TEST(boundingbox, coverage_boundingbox)
{
    bbox box1(0, 0, 99, 99);
    bbox box2(0, 0, 99, 99);
    bbox box3(0, 0, 49, 99);
    bbox box4(0, 0, 49, 49);
    bbox box5(50, 50, 99, 99);
    bbox box6(0, 0, 59, 59);
    bbox box7(0, 0, 0, 0);
    std::vector<std::tuple<bbox, bbox, float>> boxes = {
        make_tuple(box1, box2, 1),
        make_tuple(box2, box1, 1),
        make_tuple(box1, box3, 0.5),
        make_tuple(box3, box1, 1),
        make_tuple(box4, box5, 0),
        make_tuple(box5, box4, 0),
        make_tuple(box5, box6, (0.1f * 0.1f) / (0.5f * 0.5f)),
        make_tuple(box6, box5, (0.1f * 0.1f) / (0.6f * 0.6f)),
        make_tuple(bbox(), bbox(), 0),
        make_tuple(box7, box7, 1)};

    for (int i = 0; i < boxes.size(); i++)
    {
        bbox  b1    = get<0>(boxes[i]);
        bbox  b2    = get<1>(boxes[i]);
        float cover = get<2>(boxes[i]);
        EXPECT_NEAR(b1.coverage(b2), cover, nervana::epsilon) << "at index " << i;
    }
}

TEST(boundingbox, operator_equal)
{
    bbox b1(0, 1, 2, 3);
    bbox b2(0, 1, 2, 3);
    EXPECT_EQ(b1, b2);

    b2.set_xmin(10);
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.set_xmax(10);
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.set_ymin(10);
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.set_ymax(10);
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.m_label = 5;
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.m_truncated = true;
    EXPECT_NE(b1, b2);

    b2 = b1;
    ASSERT_EQ(b1, b2);
    b2.m_difficult = true;
    EXPECT_NE(b1, b2);
}
