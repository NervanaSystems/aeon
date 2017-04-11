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
#include "cpio.hpp"

#define private public

#include "interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_boundingbox.hpp"
#include "etl_label_map.hpp"
#include "json.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

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

static string read_file(const string& path)
{
    ifstream     f(path);
    stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static nlohmann::json create_box(const cv::Rect& rect, const string& label)
{
    nlohmann::json j = {{"bndbox",
                         {{"xmax", rect.x + rect.width},
                          {"xmin", rect.x},
                          {"ymax", rect.y + rect.height},
                          {"ymin", rect.y}}},
                        {"name", label}};
    return j;
}

static nlohmann::json create_metadata(const vector<nlohmann::json>& boxes, int width, int height)
{
    nlohmann::json j = nlohmann::json::object();
    j["object"]      = boxes;
    j["size"]        = {{"depth", 3}, {"height", height}, {"width", width}};
    return j;
}

static boundingbox::config make_bbox_config(int max_boxes)
{
    nlohmann::json obj = {{"height", 100}, {"width", 150}, {"max_bbox_count", max_boxes}};
    obj["class_names"] = label_list;
    return boundingbox::config(obj);
}

cv::Mat
    draw(int width, int height, const vector<boundingbox::box>& blist, cv::Rect crop = cv::Rect())
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
        string                 data = read_file(CURDIR "/test_data/000001.json");
        auto                   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(2, boxes.size());

        EXPECT_EQ(194, boxes[0].xmax);
        EXPECT_EQ(47, boxes[0].xmin);
        EXPECT_EQ(370, boxes[0].ymax);
        EXPECT_EQ(239, boxes[0].ymin);
        EXPECT_FALSE(boxes[0].difficult);
        EXPECT_TRUE(boxes[0].truncated);
        EXPECT_EQ(1, boxes[0].label);

        EXPECT_EQ(351, boxes[1].xmax);
        EXPECT_EQ(7, boxes[1].xmin);
        EXPECT_EQ(497, boxes[1].ymax);
        EXPECT_EQ(11, boxes[1].ymin);
        EXPECT_FALSE(boxes[1].difficult);
        EXPECT_TRUE(boxes[1].truncated);
        EXPECT_EQ(0, boxes[1].label);
    }
    {
        string                 data = read_file(CURDIR "/test_data/006637.json");
        auto                   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(6, boxes.size());

        auto b = boxes[3];
        EXPECT_EQ(365, b.xmax);
        EXPECT_EQ(324, b.xmin);
        EXPECT_EQ(315, b.ymax);
        EXPECT_EQ(109, b.ymin);
        EXPECT_FALSE(b.difficult);
        EXPECT_FALSE(b.truncated);
        EXPECT_EQ(0, b.label);

        auto c = b * 2.0;
        EXPECT_EQ(b.xmax * 2.0, c.xmax);

        cv::Size2i ref(500, 375);
        EXPECT_EQ(ref, decoded->image_size());

        EXPECT_EQ(3, decoded->depth());
        std::stringstream ss;
        ss << b;
        EXPECT_EQ(ss.str(), "[41 x 206 from (324, 109)] label=0");
    }
    {
        string                 data = read_file(CURDIR "/test_data/009952.json");
        auto                   cfg  = make_bbox_config(100);
        boundingbox::extractor extractor{cfg.label_map};
        auto                   decoded = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded);
        auto boxes = decoded->boxes();
        ASSERT_EQ(1, boxes.size());
    }
}

TEST(boundingbox, operator_mult)
{
    nervana::box a(1.0, 2.0, 3.0, 4.0);
    auto b = a * 4.0;
    EXPECT_EQ(4, b.x());
    EXPECT_EQ(8, b.y());
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
    string         data        = read_file(CURDIR "/test_data/009952.json");
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

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();
    ASSERT_EQ(3, boxes.size());
    EXPECT_EQ(r0, boxes[0].rect());
    EXPECT_EQ(r1, boxes[1].rect());
    EXPECT_EQ(r2, boxes[2].rect());
    EXPECT_EQ(6, boxes[0].label);
    EXPECT_EQ(8, boxes[1].label);
    EXPECT_EQ(7, boxes[2].label);

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    auto                               tx     = transform.transform(iparam, decoded);
}

TEST(boundingbox, crop)
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

    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->cropbox                           = cv::Rect(35, 35, 40, 40);

    auto d = draw(256, 256, decoded->boxes(), iparam->cropbox);
    cv::imwrite("bbox_crop.png", d);

    float scale                         = 256. / 40.;
    iparam->output_size                 = cv::Size(256, 256);
    auto                     tx_decoded = transform.transform(iparam, decoded);
    vector<boundingbox::box> tx_boxes   = tx_decoded->boxes();
    ASSERT_EQ(6, tx_boxes.size());
    EXPECT_EQ(cv::Rect((35 - 35) * scale, (35 - 35) * scale, 5 * scale, 5 * scale),
              tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect((50 - 35) * scale, (50 - 35) * scale, 10 * scale, 10 * scale),
              tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect((70 - 35) * scale, (35 - 35) * scale, 5 * scale, 5 * scale),
              tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect((35 - 35) * scale, (70 - 35) * scale, 5 * scale, 5 * scale),
              tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect((70 - 35) * scale, (70 - 35) * scale, 5 * scale, 5 * scale),
              tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect((35 - 35) * scale, (35 - 35) * scale, 40 * scale, 40 * scale),
              tx_boxes[5].rect());
}

TEST(boundingbox, rescale)
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
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->output_size                       = cv::Size(512, 1024);
    auto                     tx_decoded       = transform.transform(iparam, decoded);
    vector<boundingbox::box> tx_boxes         = tx_decoded->boxes();
    float                    xscale           = 512. / 256.;
    float                    yscale           = 1024. / 256.;
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
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->flip                              = 1;
    auto                     tx_decoded       = transform.transform(iparam, decoded);
    vector<boundingbox::box> tx_boxes         = tx_decoded->boxes();
    ASSERT_EQ(8, tx_boxes.size());

    EXPECT_EQ(cv::Rect(256 - 10 - 10 - 1, 10, 10, 10), tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 30 - 1, 30, 10, 10), tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 50 - 1, 50, 10, 10), tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 70 - 1, 30, 10, 10), tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 90 - 1, 35, 10, 10), tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 30 - 1, 70, 10, 10), tx_boxes[5].rect());
    EXPECT_EQ(cv::Rect(256 - 10 - 70 - 1, 70, 10, 10), tx_boxes[6].rect());
    EXPECT_EQ(cv::Rect(256 - 80 - 30 - 1, 30, 80, 80), tx_boxes[7].rect());
}

TEST(boundingbox, crop_flip)
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
    // cout << std::setw(4) << j << endl;

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();

    ASSERT_EQ(8, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->cropbox                           = cv::Rect(35, 35, 40, 40);
    iparam->output_size                       = cv::Size(256, 256);
    iparam->flip                              = 1;
    auto                     tx_decoded       = transform.transform(iparam, decoded);
    vector<boundingbox::box> tx_boxes         = tx_decoded->boxes();
    ASSERT_EQ(6, tx_boxes.size());

    float xscale = 256. / 40.;
    float yscale = 256. / 40.;
    EXPECT_EQ(cv::Rect((35 - 1) * xscale, 0 * yscale, 5 * xscale, 5 * yscale), tx_boxes[0].rect());
    EXPECT_EQ(cv::Rect((15 - 1) * xscale, 15 * yscale, 10 * xscale, 10 * yscale),
              tx_boxes[1].rect());
    EXPECT_EQ(cv::Rect((0 - 1) * xscale, 0 * yscale, 5 * xscale, 5 * yscale), tx_boxes[2].rect());
    EXPECT_EQ(cv::Rect((35 - 1) * xscale, 35 * yscale, 5 * xscale, 5 * yscale), tx_boxes[3].rect());
    EXPECT_EQ(cv::Rect((0 - 1) * xscale, 35 * yscale, 5 * xscale, 5 * yscale), tx_boxes[4].rect());
    EXPECT_EQ(cv::Rect((0 - 1) * xscale, 0 * yscale, 40 * xscale, 40 * yscale), tx_boxes[5].rect());
}

TEST(boundingbox, angle)
{
    // Create test metadata
    cv::Rect r0   = cv::Rect(10, 10, 10, 10);
    auto     list = {create_box(r0, "puma")};
    auto     j    = create_metadata(list, 256, 256);

    string buffer = j.dump();

    auto                     cfg = make_bbox_config(100);
    boundingbox::extractor   extractor{cfg.label_map};
    auto                     decoded = extractor.extract(&buffer[0], buffer.size());
    vector<boundingbox::box> boxes   = decoded->boxes();

    ASSERT_EQ(1, boxes.size());

    boundingbox::transformer           transform(cfg);
    shared_ptr<augment::image::params> iparam = make_params(256, 256);
    iparam->angle                             = 5;
    auto tx_decoded                           = transform.transform(iparam, decoded);
    EXPECT_EQ(nullptr, tx_decoded.get());
}

void test_values(const cv::Rect& r, float* outbuf)
{
    EXPECT_EQ(r.x, (int)outbuf[0]);
    EXPECT_EQ(r.y, (int)outbuf[1]);
    EXPECT_EQ(r.x + r.width, (int)outbuf[2]);
    EXPECT_EQ(r.y + r.height, (int)outbuf[3]);
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
    loader.load({(char*)outbuf.data()}, extracted);

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
        EXPECT_EQ(0, (int)outbuf[i]);
    }

    EXPECT_EQ(-1, (int)outbuf[outbuf.size() - 1]);
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
    loader.load({(char*)outbuf.data()}, extracted);

    test_values(r0, &outbuf[0]);
    test_values(r1, &outbuf[4]);
    test_values(r2, &outbuf[8]);
    test_values(r3, &outbuf[12]);
    test_values(r4, &outbuf[16]);
    test_values(r5, &outbuf[20]);

    EXPECT_EQ(-1, (int)outbuf[outbuf.size() - 1]);
}
