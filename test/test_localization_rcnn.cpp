/*
 Copyright 2016 Intel(R) Nervana(TM)
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
#include "gen_image.hpp"
#include "json.hpp"

#define private public

#include "file_util.hpp"
#include "etl_image.hpp"
#include "etl_localization_rcnn.hpp"
#include "json.hpp"
#include "provider_factory.hpp"
#include "log.hpp"
#include "loader.hpp"
#include "test_localization.hpp"

using namespace std;
using namespace nervana;
using namespace nervana::localization::rcnn;

vector<string> label_list = {"person",
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

vector<uint8_t> make_image_from_metadata(const std::string& metadata)
{
    nlohmann::json  js     = nlohmann::json::parse(metadata);
    int             width  = js["size"]["width"];
    int             height = js["size"]["height"];
    cv::Mat         test_image(height, width, CV_8UC3);
    vector<uint8_t> test_image_data;
    cv::imencode(".png", test_image, test_image_data);
    return test_image_data;
}

TEST(DISABLED_localization_rcnn, example)
{
    int                      height               = 1000;
    int                      width                = 1000;
    int                      batch_size           = 1;
    float                    fixed_scaling_factor = 1.6;
    std::string              manifest_root        = "/test_data";
    std::string              manifest             = "localization_manifest.tsv";
    std::vector<std::string> class_names          = {"bicycle", "person"};

    nlohmann::json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    nlohmann::json js_local = {{"type", "localization_rcnn"},
                               {"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", class_names}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", fixed_scaling_factor},
                             {"crop_enable", false},
                             {"flip_enable", true}};
    nlohmann::json config = {{"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"etl", {js_image, js_local}},
                             {"augmentation", {js_aug}}};

    loader_factory factory;
    auto           train_set = factory.get_loader(config);
}

TEST(localization_rcnn, generate_anchors)
{
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

    vector<box> expected = {{-83.0 - 1.0, -39.0 - 1.0, 100.0 - 1.0, 56.0 - 1.0},
                            {-175.0 - 1.0, -87.0 - 1.0, 192.0 - 1.0, 104.0 - 1.0},
                            {-359.0 - 1.0, -183.0 - 1.0, 376.0 - 1.0, 200.0 - 1.0},
                            {-55.0 - 1.0, -55.0 - 1.0, 72.0 - 1.0, 72.0 - 1.0},
                            {-119.0 - 1.0, -119.0 - 1.0, 136.0 - 1.0, 136.0 - 1.0},
                            {-247.0 - 1.0, -247.0 - 1.0, 264.0 - 1.0, 264.0 - 1.0},
                            {-35.0 - 1.0, -79.0 - 1.0, 52.0 - 1.0, 96.0 - 1.0},
                            {-79.0 - 1.0, -167.0 - 1.0, 96.0 - 1.0, 184.0 - 1.0},
                            {-167.0 - 1.0, -343.0 - 1.0, 184.0 - 1.0, 360.0 - 1.0}};

    // subtract 1 from the expected vector as it was generated with 1's based matlab
    //    expected -= 1;

    int            height   = 1000;
    int            width    = 1000;
    nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json js_aug   = {{"type", "image"},
                             {"flip_enable", false},
                             {"crop_enable", false},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", 1.6}};
    nlohmann::json js_loc = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};

    config cfg{js_loc};

    vector<box> actual      = anchor::generate_anchors(cfg.base_size, cfg.ratios, cfg.scales);
    vector<box> all_anchors = anchor::generate(cfg);
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_EQ(34596, all_anchors.size());
    EXPECT_EQ((9 * (62 * 62)), all_anchors.size());
    for (int i = 0; i < expected.size(); i++)
    {
        ASSERT_EQ(expected[i], actual[i]);
    }
}

void plot(const vector<box>& list, const string& prefix)
{
    float xmin = 0.0;
    float xmax = 0.0;
    float ymin = 0.0;
    float ymax = 0.0;
    for (const box& b : list)
    {
        xmin = std::min(xmin, b.xmin());
        xmax = std::max(xmax, b.xmax());
        ymin = std::min(ymin, b.ymin());
        ymax = std::max(ymax, b.ymax());
    }

    cv::Mat img(ymax - ymin, xmax - xmin, CV_8UC3);
    img = cv::Scalar(255, 255, 255);

    for (box b : list)
    {
        b.set_xmin(b.xmin() - xmin);
        b.set_xmax(b.xmax() - xmin);
        b.set_ymin(b.ymin() - ymin);
        b.set_ymax(b.ymax() - ymin);
        cv::rectangle(img, b.rect(), cv::Scalar(255, 0, 0));
    }
    box b = list[0];
    b.set_xmin(b.xmin() - xmin);
    b.set_xmax(b.xmax() - xmin);
    b.set_ymin(b.ymin() - ymin);
    b.set_ymax(b.ymax() - ymin);

    cv::rectangle(img, b.rect(), cv::Scalar(0, 0, 255));

    string fname =
        to_string(int(list[0].width())) + "x" + to_string(int(list[0].height())) + ".png";
    fname = prefix + fname;
    cv::imwrite(fname, img);
}

void plot(const string& path)
{
    string         prefix = path.substr(path.size() - 11, 6) + "-";
    string         data   = file_util::read_file_to_string(path);
    int            width  = 1000;
    int            height = 1000;
    nlohmann::json js_loc = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};

    config      cfg{js_loc};
    extractor   extractor{cfg};
    transformer transformer{cfg, -1};
    auto        extracted_metadata = extractor.extract(&data[0], data.size());
    ASSERT_NE(nullptr, extracted_metadata);
    auto                params               = make_shared<augment::image::params>();
    shared_ptr<decoded> transformed_metadata = transformer.transform(params, extracted_metadata);

    const vector<box>& an = transformer.all_anchors;

    int         last_width  = 0;
    int         last_height = 0;
    vector<box> list;
    for (const box& b : an)
    {
        if (last_width != b.width() || last_height != b.height())
        {
            if (list.size() > 0)
            {
                plot(list, prefix);
                list.clear();
            }
        }
        list.push_back(b);
        last_width  = b.width();
        last_height = b.height();
    }
    if (list.size() > 0)
    {
        plot(list, prefix);
    }

    vector<int> labels       = transformed_metadata->labels;
    vector<int> anchor_index = transformed_metadata->anchor_index;

    //    for(int i=0; i<transformed_metadata->anchor_index.size(); i++) {
    //        cout << "loader " << i << " " << transformed_metadata->anchor_index[i] << " " <<
    //        labels[transformed_metadata->anchor_index[i]] << endl;
    //        cout << an[transformed_metadata->anchor_index[i]] << endl;
    //    }

    {
        cv::Mat img(extracted_metadata->output_image_size, CV_8UC3);
        img = cv::Scalar(255, 255, 255);
        // Draw foreground boxes
        for (int i = 0; i < anchor_index.size(); i++)
        {
            int index = anchor_index[i];
            if (labels[index] == 1)
            {
                box abox = an[index];
                cv::rectangle(img, abox.rect(), cv::Scalar(0, 255, 0));
            }
        }

        // Draw bounding boxes
        for (box b : extracted_metadata->boxes())
        {
            b = b * extracted_metadata->image_scale;
            cv::rectangle(img, b.rect(), cv::Scalar(255, 0, 0));
        }
        cv::imwrite(prefix + "fg.png", img);
    }

    {
        cv::Mat img(extracted_metadata->output_image_size, CV_8UC3);
        img = cv::Scalar(255, 255, 255);
        // Draw background boxes
        for (int i = 0; i < anchor_index.size(); i++)
        {
            int index = anchor_index[i];
            if (labels[index] == 0)
            {
                box abox = an[index];
                cv::rectangle(img, abox.rect(), cv::Scalar(0, 0, 255));
            }
        }

        // Draw bounding boxes
        for (box b : extracted_metadata->boxes())
        {
            b = b * extracted_metadata->image_scale;
            cv::rectangle(img, b.rect(), cv::Scalar(255, 0, 0));
        }
        cv::imwrite(prefix + "bg.png", img);
    }
}

// TEST(DISABLED_localization,plot) {
//     plot(CURDIR"/test_data/009952.json");
// }

TEST(localization_rcnn, config)
{
    nlohmann::json js = {
        {"height", 300}, {"width", 400}, {"class_names", label_list}, {"max_gt_boxes", 100}};

    EXPECT_NO_THROW(::config cfg(js));
}

// not sure how we want to handle this error with new augmentation system
TEST(DISABLED_localization_rcnn, config_rotate)
{
    nlohmann::json js = {
        {"height", 300}, {"width", 400}, {"class_names", label_list}, {"max_gt_boxes", 100}};

    EXPECT_THROW(::config cfg(js), std::invalid_argument);
}

TEST(localization_rcnn, sample_anchors)
{
    string         data     = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    int            height   = 1000;
    int            width    = 1000;
    nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json js_loc   = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"flip_enable", false},
                             {"crop_enable", false},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", 1.6}};
    image::config                 image_config{js_image};
    config                        cfg{js_loc};
    augment::image::param_factory factory(js_aug);
    extractor                     extractor{cfg};
    transformer                   transformer{cfg, factory.fixed_scaling_factor};
    auto                          extracted_metadata = extractor.extract(&data[0], data.size());
    ASSERT_NE(nullptr, extracted_metadata);

    vector<unsigned char>              img = make_image_from_metadata(data);
    image::extractor                   ext{image_config};
    shared_ptr<image::decoded>         decoded    = ext.extract((char*)&img[0], img.size());
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);

    auto transformed_metadata = transformer.transform(params, extracted_metadata);
    ASSERT_NE(nullptr, transformed_metadata);

    vector<int>    labels       = transformed_metadata->labels;
    vector<target> bbox_targets = transformed_metadata->bbox_targets;
    vector<int>    anchor_index = transformed_metadata->anchor_index;
    vector<box>    anchors      = transformer.all_anchors;

    EXPECT_EQ(34596, labels.size());
    EXPECT_EQ(34596, bbox_targets.size());
    EXPECT_EQ(256, anchor_index.size());
    EXPECT_EQ(34596, anchors.size());

    for (int index : anchor_index)
    {
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 34596);
    }
    for (int index : anchor_index)
    {
        box b = anchors[index];
        EXPECT_GE(b.xmin(), 0);
        EXPECT_GE(b.ymin(), 0);
        EXPECT_LT(b.xmax(), cfg.width);
        EXPECT_LT(b.ymax(), cfg.height);
    }
}

TEST(localization_rcnn, transform_scale)
{
    string          metadata   = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    vector<uint8_t> image_data = make_image_from_metadata(metadata);
    int             width      = 600;
    int             height     = 600;
    nlohmann::json  js_image   = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json  js_loc     = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"flip_enable", false},
                             {"crop_enable", false},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", 1.6}};

    image::config                 image_config{js_image};
    config                        cfg{js_loc};
    augment::image::param_factory factory{js_aug};
    image::extractor              image_extractor{image_config};
    shared_ptr<image::decoded>    image_decoded =
        image_extractor.extract((const char*)image_data.data(), image_data.size());
    auto                               image_size = image_decoded->get_image_size();
    shared_ptr<augment::image::params> params =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params->debug_deterministic = true;

    extractor   extractor{cfg};
    transformer transformer{cfg, factory.fixed_scaling_factor};
    auto        decoded_data = extractor.extract(&metadata[0], metadata.size());
    ASSERT_NE(nullptr, decoded_data);

    shared_ptr<::decoded> transformed_data = transformer.transform(params, decoded_data);

    for (int i = 0; i < decoded_data->boxes().size(); i++)
    {
        boundingbox::box expected = decoded_data->boxes()[i];
        boundingbox::box actual   = transformed_data->gt_boxes[i];
        expected.set_xmin(expected.xmin() * transformed_data->image_scale);
        expected.set_ymin(expected.ymin() * transformed_data->image_scale);
        expected.set_xmax((expected.xmax() + 1) * transformed_data->image_scale - 1);
        expected.set_ymax((expected.ymax() + 1) * transformed_data->image_scale - 1);
        EXPECT_EQ(expected.xmin(), actual.xmin());
        EXPECT_EQ(expected.xmax(), actual.xmax());
        EXPECT_EQ(expected.ymin(), actual.ymin());
        EXPECT_EQ(expected.ymax(), actual.ymax());
    }
}

TEST(localization_rcnn, transform_flip)
{
    string          metadata   = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    vector<uint8_t> image_data = make_image_from_metadata(metadata);
    int             width      = 1000;
    int             height     = 1000;
    nlohmann::json  js_image   = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json  js_loc     = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"flip_enable", false},
                             {"crop_enable", false},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", 1.6}};

    image::config                 image_config{js_image};
    config                        cfg{js_loc};
    augment::image::param_factory factory{js_aug};
    image::extractor              image_extractor{image_config};

    shared_ptr<image::decoded> image_decoded =
        image_extractor.extract((const char*)image_data.data(), image_data.size());
    auto                               image_size = image_decoded->get_image_size();
    shared_ptr<augment::image::params> params =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params->debug_deterministic = true;
    params->flip                = 1;

    extractor   extractor{cfg};
    transformer transformer{cfg, factory.fixed_scaling_factor};
    auto        decoded_data = extractor.extract(&metadata[0], metadata.size());
    ASSERT_NE(nullptr, decoded_data);

    shared_ptr<decoded> transformed_data = transformer.transform(params, decoded_data);

    for (int i = 0; i < decoded_data->boxes().size(); i++)
    {
        boundingbox::box expected = decoded_data->boxes()[i];
        boundingbox::box actual   = transformed_data->gt_boxes[i];

        // flip
        auto xmin        = expected.xmin();
        int  image_width = 500;
        expected.set_xmin(image_width - expected.xmax() - 1);
        expected.set_xmax(image_width - xmin - 1);

        // scale
        float scale = 1.6;
        expected.set_xmin(expected.xmin() * scale);
        expected.set_ymin(expected.ymin() * scale);
        expected.set_xmax((expected.xmax() + 1) * scale - 1);
        expected.set_ymax((expected.ymax() + 1) * scale - 1);

        EXPECT_EQ(expected.xmin(), actual.xmin());
        EXPECT_EQ(expected.xmax(), actual.xmax());
        EXPECT_EQ(expected.ymin(), actual.ymin());
        EXPECT_EQ(expected.ymax(), actual.ymax());
    }
}

boundingbox::box crop_single_box(boundingbox::box expected, cv::Rect cropbox, float scale)
{
    expected = expected + (-cropbox.tl());

    expected.set_xmin(max<float>(expected.xmin(), 0));
    expected.set_ymin(max<float>(expected.ymin(), 0));
    expected.set_xmax(max<float>(expected.xmax(), -1));
    expected.set_ymax(max<float>(expected.ymax(), -1));

    expected.set_xmin(min<float>(expected.xmin(), cropbox.width));
    expected.set_ymin(min<float>(expected.ymin(), cropbox.height));
    expected.set_xmax(min<float>(expected.xmax(), cropbox.width - 1));
    expected.set_ymax(min<float>(expected.ymax(), cropbox.height - 1));

    expected = expected.rescale(scale, scale);

    return expected;
}

bool is_box_valid(boundingbox::box b)
{
    return b.size() > 0;
}

TEST(localization_rcnn, transform_crop)
{
    {
        string          data = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
        vector<uint8_t> test_image_data = make_image_from_metadata(data);
        int             width           = 600;
        int             height          = 600;

        nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
        nlohmann::json js_loc   = {{"width", width},
                                 {"height", height},
                                 {"class_names", label_list},
                                 {"max_gt_boxes", 64}};
        nlohmann::json js_aug = {{"type", "image"}, {"flip_enable", false}, {"scale", {0.8, 0.8}}};

        image::config                 image_config{js_image};
        config                        cfg{js_loc};
        augment::image::param_factory factory{js_aug};
        image::extractor              image_extractor{image_config};

        shared_ptr<image::decoded> image_decoded =
            image_extractor.extract((const char*)test_image_data.data(), test_image_data.size());
        auto                               image_size = image_decoded->get_image_size();
        shared_ptr<augment::image::params> params =
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);

        extractor   extractor{cfg};
        transformer transformer{cfg, factory.fixed_scaling_factor};
        auto        decoded_data = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded_data);
        shared_ptr<decoded> transformed_data = transformer.transform(params, decoded_data);

        EXPECT_EQ(6, transformed_data->gt_boxes.size());
        float scale = 2.0;
        for (int i = 0; i < transformed_data->gt_boxes.size(); i++)
        {
            boundingbox::box expected = decoded_data->boxes()[i];
            boundingbox::box actual   = transformed_data->gt_boxes[i];
            expected                  = crop_single_box(expected, params->cropbox, scale);
            EXPECT_EQ(expected, actual);
        }
    }
    {
        string          data = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
        vector<uint8_t> test_image_data = make_image_from_metadata(data);
        int             width           = 600;
        int             height          = 600;

        nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
        nlohmann::json js_loc   = {{"width", width},
                                 {"height", height},
                                 {"class_names", label_list},
                                 {"max_gt_boxes", 64}};
        nlohmann::json js_aug = {{"type", "image"}, {"flip_enable", false}, {"scale", {0.2, 0.2}}};

        image::config                 image_config{js_image};
        config                        cfg{js_loc};
        augment::image::param_factory factory{js_aug};
        image::extractor              image_extractor{image_config};

        shared_ptr<image::decoded> image_decoded =
            image_extractor.extract((const char*)test_image_data.data(), test_image_data.size());
        auto                               image_size = image_decoded->get_image_size();
        shared_ptr<augment::image::params> params =
            factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);

        extractor   extractor{cfg};
        transformer transformer{cfg, factory.fixed_scaling_factor};
        auto        decoded_data = extractor.extract(&data[0], data.size());
        ASSERT_NE(nullptr, decoded_data);
        shared_ptr<decoded> transformed_data = transformer.transform(params, decoded_data);

        vector<boundingbox::box> valid_boxes;
        float                    scale = 8.0;
        for (auto b : decoded_data->boxes())
        {
            auto cropped = crop_single_box(b, params->cropbox, scale);
            if (is_box_valid(cropped))
            {
                valid_boxes.push_back(cropped);
            }
        }

        ASSERT_EQ(valid_boxes.size(), transformed_data->gt_boxes.size());
        for (int i = 0; i < transformed_data->gt_boxes.size(); i++)
        {
            boundingbox::box expected = valid_boxes[i];
            boundingbox::box actual   = transformed_data->gt_boxes[i];
            EXPECT_EQ(expected, actual);
        }
    }
}

TEST(localization_rcnn, loader)
{
    vector<int> bbox_mask_index = {
        1200,   1262,   1324,   1386,   23954,  24016,  24078,  24090,  24140,  24152,  24202,
        24214,  24264,  24276,  24338,  24400,  24462,  24503,  24524,  24565,  24586,  24648,
        27977,  27978,  28039,  28040,  28101,  28102,  28163,  28164,  28225,  28226,  28287,
        28559,  28560,  35796,  35858,  35920,  35982,  58550,  58612,  58674,  58686,  58736,
        58748,  58798,  58810,  58860,  58872,  58934,  58996,  59058,  59099,  59120,  59161,
        59182,  59244,  62573,  62574,  62635,  62636,  62697,  62698,  62759,  62760,  62821,
        62822,  62883,  63155,  63156,  70392,  70454,  70516,  70578,  93146,  93208,  93270,
        93282,  93332,  93344,  93394,  93406,  93456,  93468,  93530,  93592,  93654,  93695,
        93716,  93757,  93778,  93840,  97169,  97170,  97231,  97232,  97293,  97294,  97355,
        97356,  97417,  97418,  97479,  97751,  97752,  104988, 105050, 105112, 105174, 127742,
        127804, 127866, 127878, 127928, 127940, 127990, 128002, 128052, 128064, 128126, 128188,
        128250, 128291, 128312, 128353, 128374, 128436, 131765, 131766, 131827, 131828, 131889,
        131890, 131951, 131952, 132013, 132014, 132075, 132347, 132348};

    map<int, float> bbox_targets = {
        {192, 2.90435},       {193, 2.81739},       {194, 2.73043},
        {195, 2.64348},       {196, 2.55652},       {197, 2.46957},
        {198, 2.38261},       {199, 2.29565},       {200, 2.2087},
        {201, 2.12174},       {202, 2.03478},       {203, 1.94783},
        {204, 1.86087},       {205, 1.77391},       {206, 1.68696},
        {207, 1.6},           {208, 1.51304},       {209, 1.42609},
        {210, 1.33913},       {211, 1.25217},       {212, 1.16522},
        {213, 1.07826},       {214, 0.991304},      {215, 0.904348},
        {216, 0.817391},      {217, 0.730435},      {218, 0.643478},
        {219, 0.556522},      {220, 0.469565},      {221, 0.382609},
        {222, 0.295652},      {223, 0.208696},      {224, 0.121739},
        {225, 0.0347827},     {226, -0.0521738},    {227, -0.13913},
        {228, -0.226087},     {229, -0.313043},     {254, 2.90435},
        {255, 2.81739},       {256, 2.73043},       {257, 2.64348},
        {258, 2.55652},       {259, 2.46957},       {260, 2.38261},
        {261, 2.29565},       {262, 2.2087},        {263, 2.12174},
        {264, 2.03478},       {265, 1.94783},       {266, 1.86087},
        {267, 1.77391},       {268, 1.68696},       {269, 1.6},
        {270, 1.51304},       {271, 1.42609},       {272, 1.33913},
        {273, 1.25217},       {274, 1.16522},       {275, 1.07826},
        {276, 0.991304},      {277, 0.904348},      {278, 0.817391},
        {279, 0.730435},      {280, 0.643478},      {281, 0.556522},
        {282, 0.469565},      {283, 0.382609},      {284, 0.295652},
        {285, 0.208696},      {286, 0.121739},      {287, 0.0347827},
        {288, -0.0521738},    {289, -0.13913},      {290, -0.226087},
        {291, -0.313043},     {316, 2.90435},       {317, 2.81739},
        {318, 2.73043},       {319, 2.64348},       {320, 2.55652},
        {321, 2.46957},       {322, 2.38261},       {323, 2.29565},
        {324, 2.2087},        {325, 2.12174},       {326, 2.03478},
        {327, 1.94783},       {328, 1.86087},       {329, 1.77391},
        {330, 1.68696},       {331, 1.6},           {332, 1.51304},
        {333, 1.42609},       {334, 1.33913},       {335, 1.25217},
        {336, 1.16522},       {337, 1.07826},       {338, 0.991304},
        {339, 0.904348},      {340, 0.817391},      {341, 0.730435},
        {342, 0.643478},      {343, 0.556522},      {344, 0.469565},
        {345, 0.382609},      {346, 0.295652},      {347, 0.208696},
        {348, 0.121739},      {349, 0.0347827},     {350, -0.0521738},
        {351, -0.13913},      {352, -0.226087},     {353, -0.313043},
        {378, 2.90435},       {379, 2.81739},       {380, 2.73043},
        {381, 2.64348},       {382, 2.55652},       {383, 2.46957},
        {384, 2.38261},       {385, 2.29565},       {386, 0.730435},
        {387, 0.643478},      {388, 0.556522},      {389, 0.469565},
        {390, 0.382609},      {391, 0.295652},      {392, 0.208696},
        {393, 0.121739},      {394, 0.0347826},     {395, -0.0521739},
        {396, -0.13913},      {397, -0.226087},     {398, -0.313044},
        {399, -0.4},          {400, -0.486957},     {401, -0.573913},
        {402, -0.66087},      {403, -0.747826},     {404, 0.643478},
        {405, 0.556522},      {406, 0.469565},      {407, 0.382609},
        {408, 0.295652},      {409, 0.208696},      {410, 0.121739},
        {411, 0.0347827},     {412, -0.0521738},    {413, -0.13913},
        {414, -0.226087},     {415, -0.313043},     {440, 2.90435},
        {441, 2.81739},       {442, 2.73043},       {443, 2.64348},
        {444, 2.55652},       {445, 2.46957},       {446, 2.38261},
        {447, 2.29565},       {448, 0.730435},      {449, 0.643478},
        {450, 0.556522},      {451, 0.469565},      {452, 0.382609},
        {453, 0.295652},      {454, 0.208696},      {455, 0.121739},
        {456, 0.0347826},     {457, -0.0521739},    {458, -0.13913},
        {459, -0.226087},     {460, -0.313044},     {461, -0.4},
        {462, -0.486957},     {463, -0.573913},     {464, -0.66087},
        {465, -0.747826},     {466, 0.643478},      {467, 0.556522},
        {468, 0.469565},      {469, 0.382609},      {470, 0.295652},
        {471, 0.208696},      {472, 0.121739},      {473, 0.0347827},
        {474, -0.0521738},    {475, -0.13913},      {476, -0.226087},
        {477, -0.313043},     {502, 0.804348},      {503, 0.717391},
        {504, 0.630435},      {505, 0.543478},      {506, 0.456522},
        {507, 0.369565},      {508, 0.282609},      {509, 0.195652},
        {510, 0.108696},      {511, 0.0217391},     {512, 0.556522},
        {513, 0.469565},      {514, 0.382609},      {515, 0.295652},
        {516, 0.208696},      {517, 0.121739},      {518, 0.0347826},
        {519, -0.0521739},    {520, -0.13913},      {521, -0.226087},
        {522, -0.313044},     {523, -0.4},          {524, -0.486957},
        {525, -0.573913},     {526, -0.66087},      {527, 0.26087},
        {528, 0.173913},      {529, 0.0869565},     {530, 0},
        {531, -0.0869565},    {532, -0.173913},     {1200, -0.0347826},
        {1262, -0.0347826},   {1324, -0.0347826},   {1386, -0.0347826},
        {23954, 0.0727272},   {24016, 0.0727272},   {24078, 0.0727272},
        {24090, 0},           {24140, 0.0727272},   {24152, 0},
        {24202, 0.0727272},   {24214, 0},           {24264, 0.0727272},
        {24276, 0},           {24338, 0},           {24400, 0},
        {24462, 0},           {24503, 0.0727274},   {24524, 0},
        {24565, 0.0727274},   {24586, 0},           {24648, 0},
        {27977, 0.0227273},   {27978, -0.0681818},  {28039, 0.0227273},
        {28040, -0.0681818},  {28101, 0.0227273},   {28102, -0.0681818},
        {28163, 0.0227273},   {28164, -0.0681818},  {28225, 0.0227273},
        {28226, -0.0681818},  {28287, 0.0227273},   {28559, 0.0363638},
        {28560, -0.0545453},  {34788, 4.23333},     {34789, 4.23333},
        {34790, 4.23333},     {34791, 4.23333},     {34792, 4.23333},
        {34793, 4.23333},     {34794, 4.23333},     {34795, 4.23333},
        {34796, 4.23333},     {34797, 4.23333},     {34798, 4.23333},
        {34799, 4.23333},     {34800, 4.23333},     {34801, 4.23333},
        {34802, 4.23333},     {34803, 4.23333},     {34804, 4.23333},
        {34805, 4.23333},     {34806, 4.23333},     {34807, 4.23333},
        {34808, 4.23333},     {34809, 4.23333},     {34810, 4.23333},
        {34811, 4.23333},     {34812, 4.23333},     {34813, 4.23333},
        {34814, 4.23333},     {34815, 4.23333},     {34816, 4.23333},
        {34817, 4.23333},     {34818, 4.23333},     {34819, 4.23333},
        {34820, 4.23333},     {34821, 4.23333},     {34822, 4.23333},
        {34823, 4.23333},     {34824, 4.23333},     {34825, 4.23333},
        {34850, 4.06667},     {34851, 4.06667},     {34852, 4.06667},
        {34853, 4.06667},     {34854, 4.06667},     {34855, 4.06667},
        {34856, 4.06667},     {34857, 4.06667},     {34858, 4.06667},
        {34859, 4.06667},     {34860, 4.06667},     {34861, 4.06667},
        {34862, 4.06667},     {34863, 4.06667},     {34864, 4.06667},
        {34865, 4.06667},     {34866, 4.06667},     {34867, 4.06667},
        {34868, 4.06667},     {34869, 4.06667},     {34870, 4.06667},
        {34871, 4.06667},     {34872, 4.06667},     {34873, 4.06667},
        {34874, 4.06667},     {34875, 4.06667},     {34876, 4.06667},
        {34877, 4.06667},     {34878, 4.06667},     {34879, 4.06667},
        {34880, 4.06667},     {34881, 4.06667},     {34882, 4.06667},
        {34883, 4.06667},     {34884, 4.06667},     {34885, 4.06667},
        {34886, 4.06667},     {34887, 4.06667},     {34912, 3.9},
        {34913, 3.9},         {34914, 3.9},         {34915, 3.9},
        {34916, 3.9},         {34917, 3.9},         {34918, 3.9},
        {34919, 3.9},         {34920, 3.9},         {34921, 3.9},
        {34922, 3.9},         {34923, 3.9},         {34924, 3.9},
        {34925, 3.9},         {34926, 3.9},         {34927, 3.9},
        {34928, 3.9},         {34929, 3.9},         {34930, 3.9},
        {34931, 3.9},         {34932, 3.9},         {34933, 3.9},
        {34934, 3.9},         {34935, 3.9},         {34936, 3.9},
        {34937, 3.9},         {34938, 3.9},         {34939, 3.9},
        {34940, 3.9},         {34941, 3.9},         {34942, 3.9},
        {34943, 3.9},         {34944, 3.9},         {34945, 3.9},
        {34946, 3.9},         {34947, 3.9},         {34948, 3.9},
        {34949, 3.9},         {34974, 3.73333},     {34975, 3.73333},
        {34976, 3.73333},     {34977, 3.73333},     {34978, 3.73333},
        {34979, 3.73333},     {34980, 3.73333},     {34981, 3.73333},
        {34982, 1.75833},     {34983, 1.75833},     {34984, 1.75833},
        {34985, 1.75833},     {34986, 1.75833},     {34987, 1.75833},
        {34988, 1.75833},     {34989, 1.75833},     {34990, 1.75833},
        {34991, 1.75833},     {34992, 1.75833},     {34993, 1.75833},
        {34994, 1.75833},     {34995, 1.75833},     {34996, 1.75833},
        {34997, 1.75833},     {34998, 1.75833},     {34999, 1.75833},
        {35000, 3.73333},     {35001, 3.73333},     {35002, 3.73333},
        {35003, 3.73333},     {35004, 3.73333},     {35005, 3.73333},
        {35006, 3.73333},     {35007, 3.73333},     {35008, 3.73333},
        {35009, 3.73333},     {35010, 3.73333},     {35011, 3.73333},
        {35036, 3.56667},     {35037, 3.56667},     {35038, 3.56667},
        {35039, 3.56667},     {35040, 3.56667},     {35041, 3.56667},
        {35042, 3.56667},     {35043, 3.56667},     {35044, 1.59167},
        {35045, 1.59167},     {35046, 1.59167},     {35047, 1.59167},
        {35048, 1.59167},     {35049, 1.59167},     {35050, 1.59167},
        {35051, 1.59167},     {35052, 1.59167},     {35053, 1.59167},
        {35054, 1.59167},     {35055, 1.59167},     {35056, 1.59167},
        {35057, 1.59167},     {35058, 1.59167},     {35059, 1.59167},
        {35060, 1.59167},     {35061, 1.59167},     {35062, 3.56667},
        {35063, 3.56667},     {35064, 3.56667},     {35065, 3.56667},
        {35066, 3.56667},     {35067, 3.56667},     {35068, 3.56667},
        {35069, 3.56667},     {35070, 3.56667},     {35071, 3.56667},
        {35072, 3.56667},     {35073, 3.56667},     {35098, 1.86667},
        {35099, 1.86667},     {35100, 1.86667},     {35101, 1.86667},
        {35102, 1.86667},     {35103, 1.86667},     {35104, 1.86667},
        {35105, 1.86667},     {35106, 1.86667},     {35107, 1.86667},
        {35108, 1.425},       {35109, 1.425},       {35110, 1.425},
        {35111, 1.425},       {35112, 1.425},       {35113, 1.425},
        {35114, 1.425},       {35115, 1.425},       {35116, 1.425},
        {35117, 1.425},       {35118, 1.425},       {35119, 1.425},
        {35120, 1.425},       {35121, 1.425},       {35122, 1.425},
        {35123, 2.125},       {35124, 2.125},       {35125, 2.125},
        {35126, 2.125},       {35127, 2.125},       {35128, 2.125},
        {35796, 0.216667},    {35858, 0.0499999},   {35920, -0.116667},
        {35982, -0.283333},   {58550, 0.231818},    {58612, 0.140909},
        {58674, 0.0499999},   {58686, 0.431818},    {58736, -0.0409092},
        {58748, 0.340909},    {58798, -0.131818},   {58810, 0.25},
        {58860, -0.222727},   {58872, 0.159091},    {58934, 0.0681818},
        {58996, -0.0227273},  {59058, -0.113636},   {59099, 0.0318182},
        {59120, -0.204545},   {59161, -0.0590909},  {59182, -0.295455},
        {59244, -0.386364},   {62573, 0.1},         {62574, 0.1},
        {62635, 0.0545455},   {62636, 0.0545455},   {62697, 0.00909094},
        {62698, 0.00909094},  {62759, -0.0363636},  {62760, -0.0363636},
        {62821, -0.0818181},  {62822, -0.0818181},  {62883, -0.127273},
        {63155, 0.109091},    {63156, 0.109091},    {69384, -0.180584},
        {69385, -0.180584},   {69386, -0.180584},   {69387, -0.180584},
        {69388, -0.180584},   {69389, -0.180584},   {69390, -0.180584},
        {69391, -0.180584},   {69392, -0.180584},   {69393, -0.180584},
        {69394, -0.180584},   {69395, -0.180584},   {69396, -0.180584},
        {69397, -0.180584},   {69398, -0.180584},   {69399, -0.180584},
        {69400, -0.180584},   {69401, -0.180584},   {69402, -0.180584},
        {69403, -0.180584},   {69404, -0.180584},   {69405, -0.180584},
        {69406, -0.180584},   {69407, -0.180584},   {69408, -0.180584},
        {69409, -0.180584},   {69410, -0.180584},   {69411, -0.180584},
        {69412, -0.180584},   {69413, -0.180584},   {69414, -0.180584},
        {69415, -0.180584},   {69416, -0.180584},   {69417, -0.180584},
        {69418, -0.180584},   {69419, -0.180584},   {69420, -0.180584},
        {69421, -0.180584},   {69446, -0.180584},   {69447, -0.180584},
        {69448, -0.180584},   {69449, -0.180584},   {69450, -0.180584},
        {69451, -0.180584},   {69452, -0.180584},   {69453, -0.180584},
        {69454, -0.180584},   {69455, -0.180584},   {69456, -0.180584},
        {69457, -0.180584},   {69458, -0.180584},   {69459, -0.180584},
        {69460, -0.180584},   {69461, -0.180584},   {69462, -0.180584},
        {69463, -0.180584},   {69464, -0.180584},   {69465, -0.180584},
        {69466, -0.180584},   {69467, -0.180584},   {69468, -0.180584},
        {69469, -0.180584},   {69470, -0.180584},   {69471, -0.180584},
        {69472, -0.180584},   {69473, -0.180584},   {69474, -0.180584},
        {69475, -0.180584},   {69476, -0.180584},   {69477, -0.180584},
        {69478, -0.180584},   {69479, -0.180584},   {69480, -0.180584},
        {69481, -0.180584},   {69482, -0.180584},   {69483, -0.180584},
        {69508, -0.180584},   {69509, -0.180584},   {69510, -0.180584},
        {69511, -0.180584},   {69512, -0.180584},   {69513, -0.180584},
        {69514, -0.180584},   {69515, -0.180584},   {69516, -0.180584},
        {69517, -0.180584},   {69518, -0.180584},   {69519, -0.180584},
        {69520, -0.180584},   {69521, -0.180584},   {69522, -0.180584},
        {69523, -0.180584},   {69524, -0.180584},   {69525, -0.180584},
        {69526, -0.180584},   {69527, -0.180584},   {69528, -0.180584},
        {69529, -0.180584},   {69530, -0.180584},   {69531, -0.180584},
        {69532, -0.180584},   {69533, -0.180584},   {69534, -0.180584},
        {69535, -0.180584},   {69536, -0.180584},   {69537, -0.180584},
        {69538, -0.180584},   {69539, -0.180584},   {69540, -0.180584},
        {69541, -0.180584},   {69542, -0.180584},   {69543, -0.180584},
        {69544, -0.180584},   {69545, -0.180584},   {69570, -0.180584},
        {69571, -0.180584},   {69572, -0.180584},   {69573, -0.180584},
        {69574, -0.180584},   {69575, -0.180584},   {69576, -0.180584},
        {69577, -0.180584},   {69578, -0.650588},   {69579, -0.650588},
        {69580, -0.650588},   {69581, -0.650588},   {69582, -0.650588},
        {69583, -0.650588},   {69584, -0.650588},   {69585, -0.650588},
        {69586, -0.650588},   {69587, -0.650588},   {69588, -0.650588},
        {69589, -0.650588},   {69590, -0.650588},   {69591, -0.650588},
        {69592, -0.650588},   {69593, -0.650588},   {69594, -0.650588},
        {69595, -0.650588},   {69596, -0.180584},   {69597, -0.180584},
        {69598, -0.180584},   {69599, -0.180584},   {69600, -0.180584},
        {69601, -0.180584},   {69602, -0.180584},   {69603, -0.180584},
        {69604, -0.180584},   {69605, -0.180584},   {69606, -0.180584},
        {69607, -0.180584},   {69632, -0.180584},   {69633, -0.180584},
        {69634, -0.180584},   {69635, -0.180584},   {69636, -0.180584},
        {69637, -0.180584},   {69638, -0.180584},   {69639, -0.180584},
        {69640, -0.650588},   {69641, -0.650588},   {69642, -0.650588},
        {69643, -0.650588},   {69644, -0.650588},   {69645, -0.650588},
        {69646, -0.650588},   {69647, -0.650588},   {69648, -0.650588},
        {69649, -0.650588},   {69650, -0.650588},   {69651, -0.650588},
        {69652, -0.650588},   {69653, -0.650588},   {69654, -0.650588},
        {69655, -0.650588},   {69656, -0.650588},   {69657, -0.650588},
        {69658, -0.180584},   {69659, -0.180584},   {69660, -0.180584},
        {69661, -0.180584},   {69662, -0.180584},   {69663, -0.180584},
        {69664, -0.180584},   {69665, -0.180584},   {69666, -0.180584},
        {69667, -0.180584},   {69668, -0.180584},   {69669, -0.180584},
        {69694, -0.0909718},  {69695, -0.0909718},  {69696, -0.0909718},
        {69697, -0.0909718},  {69698, -0.0909718},  {69699, -0.0909718},
        {69700, -0.0909718},  {69701, -0.0909718},  {69702, -0.0909718},
        {69703, -0.0909718},  {69704, -0.650588},   {69705, -0.650588},
        {69706, -0.650588},   {69707, -0.650588},   {69708, -0.650588},
        {69709, -0.650588},   {69710, -0.650588},   {69711, -0.650588},
        {69712, -0.650588},   {69713, -0.650588},   {69714, -0.650588},
        {69715, -0.650588},   {69716, -0.650588},   {69717, -0.650588},
        {69718, -0.650588},   {69719, -1.00726},    {69720, -1.00726},
        {69721, -1.00726},    {69722, -1.00726},    {69723, -1.00726},
        {69724, -1.00726},    {70392, -0.00873357}, {70454, -0.00873357},
        {70516, -0.00873357}, {70578, -0.00873357}, {93146, 0.0870114},
        {93208, 0.0870114},   {93270, 0.0870114},   {93282, -0.269663},
        {93332, 0.0870114},   {93344, -0.269663},   {93394, 0.0870114},
        {93406, -0.269663},   {93456, 0.0870114},   {93468, -0.269663},
        {93530, -0.269663},   {93592, -0.269663},   {93654, -0.269663},
        {93695, 0.182322},    {93716, -0.269663},   {93757, 0.182322},
        {93778, -0.269663},   {93840, -0.269663},   {97169, -0.04652},
        {97170, -0.04652},    {97231, -0.04652},    {97232, -0.04652},
        {97293, -0.04652},    {97294, -0.04652},    {97355, -0.04652},
        {97356, -0.04652},    {97417, -0.04652},    {97418, -0.04652},
        {97479, -0.04652},    {97751, -0.136132},   {97752, -0.136132},
        {103980, 1.01764},    {103981, 1.01764},    {103982, 1.01764},
        {103983, 1.01764},    {103984, 1.01764},    {103985, 1.01764},
        {103986, 1.01764},    {103987, 1.01764},    {103988, 1.01764},
        {103989, 1.01764},    {103990, 1.01764},    {103991, 1.01764},
        {103992, 1.01764},    {103993, 1.01764},    {103994, 1.01764},
        {103995, 1.01764},    {103996, 1.01764},    {103997, 1.01764},
        {103998, 1.01764},    {103999, 1.01764},    {104000, 1.01764},
        {104001, 1.01764},    {104002, 1.01764},    {104003, 1.01764},
        {104004, 1.01764},    {104005, 1.01764},    {104006, 1.01764},
        {104007, 1.01764},    {104008, 1.01764},    {104009, 1.01764},
        {104010, 1.01764},    {104011, 1.01764},    {104012, 1.01764},
        {104013, 1.01764},    {104014, 1.01764},    {104015, 1.01764},
        {104016, 1.01764},    {104017, 1.01764},    {104042, 1.01764},
        {104043, 1.01764},    {104044, 1.01764},    {104045, 1.01764},
        {104046, 1.01764},    {104047, 1.01764},    {104048, 1.01764},
        {104049, 1.01764},    {104050, 1.01764},    {104051, 1.01764},
        {104052, 1.01764},    {104053, 1.01764},    {104054, 1.01764},
        {104055, 1.01764},    {104056, 1.01764},    {104057, 1.01764},
        {104058, 1.01764},    {104059, 1.01764},    {104060, 1.01764},
        {104061, 1.01764},    {104062, 1.01764},    {104063, 1.01764},
        {104064, 1.01764},    {104065, 1.01764},    {104066, 1.01764},
        {104067, 1.01764},    {104068, 1.01764},    {104069, 1.01764},
        {104070, 1.01764},    {104071, 1.01764},    {104072, 1.01764},
        {104073, 1.01764},    {104074, 1.01764},    {104075, 1.01764},
        {104076, 1.01764},    {104077, 1.01764},    {104078, 1.01764},
        {104079, 1.01764},    {104104, 1.01764},    {104105, 1.01764},
        {104106, 1.01764},    {104107, 1.01764},    {104108, 1.01764},
        {104109, 1.01764},    {104110, 1.01764},    {104111, 1.01764},
        {104112, 1.01764},    {104113, 1.01764},    {104114, 1.01764},
        {104115, 1.01764},    {104116, 1.01764},    {104117, 1.01764},
        {104118, 1.01764},    {104119, 1.01764},    {104120, 1.01764},
        {104121, 1.01764},    {104122, 1.01764},    {104123, 1.01764},
        {104124, 1.01764},    {104125, 1.01764},    {104126, 1.01764},
        {104127, 1.01764},    {104128, 1.01764},    {104129, 1.01764},
        {104130, 1.01764},    {104131, 1.01764},    {104132, 1.01764},
        {104133, 1.01764},    {104134, 1.01764},    {104135, 1.01764},
        {104136, 1.01764},    {104137, 1.01764},    {104138, 1.01764},
        {104139, 1.01764},    {104140, 1.01764},    {104141, 1.01764},
        {104166, 1.01764},    {104167, 1.01764},    {104168, 1.01764},
        {104169, 1.01764},    {104170, 1.01764},    {104171, 1.01764},
        {104172, 1.01764},    {104173, 1.01764},    {104174, 1.03555},
        {104175, 1.03555},    {104176, 1.03555},    {104177, 1.03555},
        {104178, 1.03555},    {104179, 1.03555},    {104180, 1.03555},
        {104181, 1.03555},    {104182, 1.03555},    {104183, 1.03555},
        {104184, 1.03555},    {104185, 1.03555},    {104186, 1.03555},
        {104187, 1.03555},    {104188, 1.03555},    {104189, 1.03555},
        {104190, 1.03555},    {104191, 1.03555},    {104192, 1.01764},
        {104193, 1.01764},    {104194, 1.01764},    {104195, 1.01764},
        {104196, 1.01764},    {104197, 1.01764},    {104198, 1.01764},
        {104199, 1.01764},    {104200, 1.01764},    {104201, 1.01764},
        {104202, 1.01764},    {104203, 1.01764},    {104228, 1.01764},
        {104229, 1.01764},    {104230, 1.01764},    {104231, 1.01764},
        {104232, 1.01764},    {104233, 1.01764},    {104234, 1.01764},
        {104235, 1.01764},    {104236, 1.03555},    {104237, 1.03555},
        {104238, 1.03555},    {104239, 1.03555},    {104240, 1.03555},
        {104241, 1.03555},    {104242, 1.03555},    {104243, 1.03555},
        {104244, 1.03555},    {104245, 1.03555},    {104246, 1.03555},
        {104247, 1.03555},    {104248, 1.03555},    {104249, 1.03555},
        {104250, 1.03555},    {104251, 1.03555},    {104252, 1.03555},
        {104253, 1.03555},    {104254, 1.01764},    {104255, 1.01764},
        {104256, 1.01764},    {104257, 1.01764},    {104258, 1.01764},
        {104259, 1.01764},    {104260, 1.01764},    {104261, 1.01764},
        {104262, 1.01764},    {104263, 1.01764},    {104264, 1.01764},
        {104265, 1.01764},    {104290, 1.10966},    {104291, 1.10966},
        {104292, 1.10966},    {104293, 1.10966},    {104294, 1.10966},
        {104295, 1.10966},    {104296, 1.10966},    {104297, 1.10966},
        {104298, 1.10966},    {104299, 1.10966},    {104300, 1.03555},
        {104301, 1.03555},    {104302, 1.03555},    {104303, 1.03555},
        {104304, 1.03555},    {104305, 1.03555},    {104306, 1.03555},
        {104307, 1.03555},    {104308, 1.03555},    {104309, 1.03555},
        {104310, 1.03555},    {104311, 1.03555},    {104312, 1.03555},
        {104313, 1.03555},    {104314, 1.03555},    {104315, 1.23837},
        {104316, 1.23837},    {104317, 1.23837},    {104318, 1.23837},
        {104319, 1.23837},    {104320, 1.23837},    {104988, 0.530628},
        {105050, 0.530628},   {105112, 0.530628},   {105174, 0.530628},
        {127742, 0.429418},   {127804, 0.429418},   {127866, 0.429418},
        {127878, 0.632239},   {127928, 0.429418},   {127940, 0.632239},
        {127990, 0.429418},   {128002, 0.632239},   {128052, 0.429418},
        {128064, 0.632239},   {128126, 0.632239},   {128188, 0.632239},
        {128250, 0.632239},   {128291, 0.174717},   {128312, 0.632239},
        {128353, 0.174717},   {128374, 0.632239},   {128436, 0.632239},
        {131765, -0.189621},  {131766, -0.189621},  {131827, -0.189621},
        {131828, -0.189621},  {131889, -0.189621},  {131890, -0.189621},
        {131951, -0.189621},  {131952, -0.189621},  {132013, -0.189621},
        {132014, -0.189621},  {132075, -0.189621},  {132347, -0.28164},
        {132348, -0.28164},
    };

    // These two tables were generated with model private-neon/examples/rpn
    // random choice was disabled
    vector<int> fg_idx = {1200,  1262,  1324,  1386,  23954, 24016, 24078, 24090, 24140,
                          24152, 24202, 24214, 24264, 24276, 24338, 24400, 24462, 24503,
                          24524, 24565, 24586, 24648, 27977, 27978, 28039, 28040, 28101,
                          28102, 28163, 28164, 28225, 28226, 28287, 28559, 28560};

    vector<int> bg_idx = {
        192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
        209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
        226, 227, 228, 229, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
        267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
        284, 285, 286, 287, 288, 289, 290, 291, 316, 317, 318, 319, 320, 321, 322, 323, 324,
        325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341,
        342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 378, 379, 380, 381, 382,
        383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,
        400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 440,
        441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
        458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
        475, 476, 477, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
        516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532};

    // bg and fg
    vector<int> labels_mask_idx = {
        192,   193,   194,   195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
        205,   206,   207,   208,   209,   210,   211,   212,   213,   214,   215,   216,   217,
        218,   219,   220,   221,   222,   223,   224,   225,   226,   227,   228,   229,   254,
        255,   256,   257,   258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
        268,   269,   270,   271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
        281,   282,   283,   284,   285,   286,   287,   288,   289,   290,   291,   316,   317,
        318,   319,   320,   321,   322,   323,   324,   325,   326,   327,   328,   329,   330,
        331,   332,   333,   334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
        344,   345,   346,   347,   348,   349,   350,   351,   352,   353,   378,   379,   380,
        381,   382,   383,   384,   385,   386,   387,   388,   389,   390,   391,   392,   393,
        394,   395,   396,   397,   398,   399,   400,   401,   402,   403,   404,   405,   406,
        407,   408,   409,   410,   411,   412,   413,   414,   415,   440,   441,   442,   443,
        444,   445,   446,   447,   448,   449,   450,   451,   452,   453,   454,   455,   456,
        457,   458,   459,   460,   461,   462,   463,   464,   465,   466,   467,   468,   469,
        470,   471,   472,   473,   474,   475,   476,   477,   502,   503,   504,   505,   506,
        507,   508,   509,   510,   511,   512,   513,   514,   515,   516,   517,   518,   519,
        520,   521,   522,   523,   524,   525,   526,   527,   528,   529,   530,   531,   532,
        1200,  1262,  1324,  1386,  23954, 24016, 24078, 24090, 24140, 24152, 24202, 24214, 24264,
        24276, 24338, 24400, 24462, 24503, 24524, 24565, 24586, 24648, 27977, 27978, 28039, 28040,
        28101, 28102, 28163, 28164, 28225, 28226, 28287, 28559, 28560, 34788, 34789, 34790, 34791,
        34792, 34793, 34794, 34795, 34796, 34797, 34798, 34799, 34800, 34801, 34802, 34803, 34804,
        34805, 34806, 34807, 34808, 34809, 34810, 34811, 34812, 34813, 34814, 34815, 34816, 34817,
        34818, 34819, 34820, 34821, 34822, 34823, 34824, 34825, 34850, 34851, 34852, 34853, 34854,
        34855, 34856, 34857, 34858, 34859, 34860, 34861, 34862, 34863, 34864, 34865, 34866, 34867,
        34868, 34869, 34870, 34871, 34872, 34873, 34874, 34875, 34876, 34877, 34878, 34879, 34880,
        34881, 34882, 34883, 34884, 34885, 34886, 34887, 34912, 34913, 34914, 34915, 34916, 34917,
        34918, 34919, 34920, 34921, 34922, 34923, 34924, 34925, 34926, 34927, 34928, 34929, 34930,
        34931, 34932, 34933, 34934, 34935, 34936, 34937, 34938, 34939, 34940, 34941, 34942, 34943,
        34944, 34945, 34946, 34947, 34948, 34949, 34974, 34975, 34976, 34977, 34978, 34979, 34980,
        34981, 34982, 34983, 34984, 34985, 34986, 34987, 34988, 34989, 34990, 34991, 34992, 34993,
        34994, 34995, 34996, 34997, 34998, 34999, 35000, 35001, 35002, 35003, 35004, 35005, 35006,
        35007, 35008, 35009, 35010, 35011, 35036, 35037, 35038, 35039, 35040, 35041, 35042, 35043,
        35044, 35045, 35046, 35047, 35048, 35049, 35050, 35051, 35052, 35053, 35054, 35055, 35056,
        35057, 35058, 35059, 35060, 35061, 35062, 35063, 35064, 35065, 35066, 35067, 35068, 35069,
        35070, 35071, 35072, 35073, 35098, 35099, 35100, 35101, 35102, 35103, 35104, 35105, 35106,
        35107, 35108, 35109, 35110, 35111, 35112, 35113, 35114, 35115, 35116, 35117, 35118, 35119,
        35120, 35121, 35122, 35123, 35124, 35125, 35126, 35127, 35128, 35796, 35858, 35920, 35982,
        58550, 58612, 58674, 58686, 58736, 58748, 58798, 58810, 58860, 58872, 58934, 58996, 59058,
        59099, 59120, 59161, 59182, 59244, 62573, 62574, 62635, 62636, 62697, 62698, 62759, 62760,
        62821, 62822, 62883, 63155, 63156};

    string         data     = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    int            width    = 1000;
    int            height   = 1000;
    nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json js_loc   = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"flip_enable", false},
                             {"crop_enable", false},
                             {"fixed_aspect_ratio", true},
                             {"fixed_scaling_factor", 1.6}};
    image::config                 image_config{js_image};
    config                        cfg{js_loc};
    augment::image::param_factory factory(js_aug);

    extractor                  extractor{cfg};
    transformer                transformer{cfg, factory.fixed_scaling_factor};
    localization::rcnn::loader loader{cfg};
    auto                       extract_data = extractor.extract(&data[0], data.size());
    ASSERT_NE(nullptr, extract_data);

    vector<unsigned char>              img = make_image_from_metadata(data);
    image::extractor                   ext{image_config};
    shared_ptr<image::decoded>         decoded    = ext.extract((char*)&img[0], img.size());
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params->debug_deterministic = true;

    shared_ptr<::decoded> transformed_data = transformer.transform(params, extract_data);

    ASSERT_EQ(transformed_data->anchor_index.size(), fg_idx.size() + bg_idx.size());
    for (int i = 0; i < fg_idx.size(); i++)
    {
        ASSERT_EQ(fg_idx[i], transformed_data->anchor_index[i]);
    }
    for (int i = 0; i < bg_idx.size(); i++)
    {
        ASSERT_EQ(bg_idx[i], transformed_data->anchor_index[i + fg_idx.size()]);
    }

    vector<float>   bbtargets;
    vector<float>   bbtargets_mask;
    vector<int32_t> labels_flat;
    vector<int32_t> labels_mask;
    vector<int32_t> im_shape;
    vector<float>   gt_boxes;
    vector<int32_t> num_gt_boxes;
    vector<int32_t> gt_classes;
    vector<float>   im_scale;
    vector<int32_t> gt_difficult;

    vector<void*>             buf_list;
    const vector<shape_type>& shapes = cfg.get_shape_type_list();
    ASSERT_EQ(10, shapes.size());
    size_t total_anchors = 34596;

    bbtargets.resize(shapes[0].get_element_count());
    bbtargets_mask.resize(shapes[1].get_element_count());
    labels_flat.resize(shapes[2].get_element_count());
    labels_mask.resize(shapes[3].get_element_count());
    im_shape.resize(shapes[4].get_element_count());
    gt_boxes.resize(shapes[5].get_element_count());
    num_gt_boxes.resize(shapes[6].get_element_count());
    gt_classes.resize(shapes[7].get_element_count());
    im_scale.resize(shapes[8].get_element_count());
    gt_difficult.resize(shapes[9].get_element_count());

    ASSERT_EQ(total_anchors * 4, bbtargets.size());
    ASSERT_EQ(total_anchors * 4, bbtargets_mask.size());
    ASSERT_EQ(total_anchors * 2, labels_flat.size());
    ASSERT_EQ(total_anchors * 2, labels_mask.size());
    ASSERT_EQ(2, im_shape.size());
    ASSERT_EQ(64 * 4, gt_boxes.size());
    ASSERT_EQ(1, num_gt_boxes.size());
    ASSERT_EQ(64, gt_classes.size());
    ASSERT_EQ(1, im_scale.size());
    ASSERT_EQ(64, gt_difficult.size());

    memset(bbtargets.data(), 0xFF, bbtargets.size() * sizeof(float));
    memset(bbtargets_mask.data(), 0xFF, bbtargets_mask.size() * sizeof(float));
    memset(labels_flat.data(), 0xFF, labels_flat.size() * sizeof(int32_t));
    memset(labels_mask.data(), 0xFF, labels_mask.size() * sizeof(int32_t));
    memset(im_shape.data(), 0xFF, im_shape.size() * sizeof(int32_t));
    memset(gt_boxes.data(), 0xFF, gt_boxes.size() * sizeof(float));
    memset(num_gt_boxes.data(), 0xFF, num_gt_boxes.size() * sizeof(int32_t));
    memset(gt_classes.data(), 0xFF, gt_classes.size() * sizeof(int32_t));
    memset(im_scale.data(), 0xFF, im_scale.size() * sizeof(float));
    memset(gt_difficult.data(), 0xFF, gt_difficult.size() * sizeof(int32_t));

    buf_list.push_back(bbtargets.data());
    buf_list.push_back(bbtargets_mask.data());
    buf_list.push_back(labels_flat.data());
    buf_list.push_back(labels_mask.data());
    buf_list.push_back(im_shape.data());
    buf_list.push_back(gt_boxes.data());
    buf_list.push_back(num_gt_boxes.data());
    buf_list.push_back(gt_classes.data());
    buf_list.push_back(im_scale.data());
    buf_list.push_back(gt_difficult.data());

    loader.load(buf_list, transformed_data);

    //    loader.build_output(transformed_data, labels, labels_mask, bbtargets, bbtargets_mask);

    //-------------------------------------------------------------------------
    // labels
    //-------------------------------------------------------------------------
    for (size_t i = 0; i < labels_flat.size() / 2; i++)
    {
        auto p = find(fg_idx.begin(), fg_idx.end(), i);
        if (p != fg_idx.end())
        {
            ASSERT_EQ(0, labels_flat[i]) << "at index " << i;
            ASSERT_EQ(1, labels_flat[i + total_anchors]) << "at index " << i;
        }
        else
        {
            ASSERT_EQ(1, labels_flat[i]) << "at index " << i;
            ASSERT_EQ(0, labels_flat[i + total_anchors]) << "at index " << i;
        }
    }

    //-------------------------------------------------------------------------
    // labels_mask
    //-------------------------------------------------------------------------
    for (size_t i = 0; i < labels_mask.size(); i++)
    {
        auto p = find(labels_mask_idx.begin(), labels_mask_idx.end(), i);
        if (p != labels_mask_idx.end())
        {
            ASSERT_EQ(1, labels_mask[i]) << "at index " << i;
        }
        else
        {
            ASSERT_EQ(0, labels_mask[i]) << "at index " << i;
        }
    }

    //-------------------------------------------------------------------------
    // bbtargets
    //-------------------------------------------------------------------------
    for (int i = 0; i < bbtargets.size(); i++)
    {
        auto p = bbox_targets.find(i);
        if (p != bbox_targets.end())
        {
            ASSERT_NEAR(p->second, bbtargets[i], 0.00001) << "at index " << i;
        }
        else
        {
            ASSERT_EQ(0., bbtargets[i]) << "at index " << i;
        }
    }

    //-------------------------------------------------------------------------
    // bbtargets_mask
    //-------------------------------------------------------------------------
    for (int i = 0; i < bbtargets_mask.size() / 4; i++)
    {
        auto fg = find(fg_idx.begin(), fg_idx.end(), i);
        if (fg != fg_idx.end())
        {
            ASSERT_EQ(1, bbtargets_mask[i + total_anchors * 0]) << "at index " << i;
            ASSERT_EQ(1, bbtargets_mask[i + total_anchors * 1]) << "at index " << i;
            ASSERT_EQ(1, bbtargets_mask[i + total_anchors * 2]) << "at index " << i;
            ASSERT_EQ(1, bbtargets_mask[i + total_anchors * 3]) << "at index " << i;
        }
        else
        {
            ASSERT_EQ(0, bbtargets_mask[i + total_anchors * 0]) << "at index " << i;
            ASSERT_EQ(0, bbtargets_mask[i + total_anchors * 1]) << "at index " << i;
            ASSERT_EQ(0, bbtargets_mask[i + total_anchors * 2]) << "at index " << i;
            ASSERT_EQ(0, bbtargets_mask[i + total_anchors * 3]) << "at index " << i;
        }
    }

    EXPECT_EQ(800, im_shape[0]) << "width";
    EXPECT_EQ(600, im_shape[1]) << "height";
    EXPECT_EQ(6, num_gt_boxes[0]);
    for (int i = 0; i < num_gt_boxes[0]; i++)
    {
        const boundingbox::box& box = transformed_data->boxes()[i];
        EXPECT_EQ(box.xmin() * im_scale[0], gt_boxes[i * 4 + 0]);
        EXPECT_EQ(box.ymin() * im_scale[0], gt_boxes[i * 4 + 1]);
        EXPECT_EQ((box.xmax() + 1) * im_scale[0] - 1, gt_boxes[i * 4 + 2]);
        EXPECT_EQ((box.ymax() + 1) * im_scale[0] - 1, gt_boxes[i * 4 + 3]);
        EXPECT_EQ(box.label(), gt_classes[i]);
        EXPECT_EQ(box.difficult(), gt_difficult[i]);
    }
    EXPECT_FLOAT_EQ(1.6, im_scale[0]);
}

TEST(localization_rcnn, loader_zero_gt_boxes)
{
    string data   = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    int    width  = 1000;
    int    height = 1000;

    nlohmann::json js_image = {{"width", width}, {"height", height}, {"channels", 3}};
    nlohmann::json js_loc   = {
        {"width", width}, {"height", height}, {"class_names", label_list}, {"max_gt_boxes", 64}};
    nlohmann::json js_aug = {{"type", "image"}, {"flip_enable", false}, {"scale", {0.1, 0.1}}};
    image::config  image_config{js_image};
    config         cfg{js_loc};
    augment::image::param_factory factory(js_aug);

    extractor                  extractor{cfg};
    transformer                transformer{cfg, factory.fixed_scaling_factor};
    localization::rcnn::loader loader{cfg};
    auto                       extract_data = extractor.extract(&data[0], data.size());
    ASSERT_NE(nullptr, extract_data);

    vector<unsigned char>              img = make_image_from_metadata(data);
    image::extractor                   ext{image_config};
    shared_ptr<image::decoded>         decoded    = ext.extract((char*)&img[0], img.size());
    auto                               image_size = decoded->get_image_size();
    shared_ptr<augment::image::params> params =
        factory.make_params(image_size.width, image_size.height, cfg.width, cfg.height);
    params->debug_deterministic = true;
    params->cropbox.x           = 0;
    params->cropbox.y           = 0;

    shared_ptr<::decoded> transformed_data = transformer.transform(params, extract_data);
    ASSERT_EQ(0, transformed_data->gt_boxes.size());

    //    ASSERT_EQ(transformed_data->anchor_index.size(), fg_idx.size() + bg_idx.size());
    //    for(int i=0; i<fg_idx.size(); i++) {
    //        ASSERT_EQ(fg_idx[i], transformed_data->anchor_index[i]);
    //    }
    //    for(int i=0; i<bg_idx.size(); i++) {
    //        ASSERT_EQ(bg_idx[i], transformed_data->anchor_index[i+fg_idx.size()]);
    //    }

    vector<float>   bbtargets;
    vector<float>   bbtargets_mask;
    vector<int32_t> labels_flat;
    vector<int32_t> labels_mask;
    vector<int32_t> im_shape;
    vector<float>   gt_boxes;
    vector<int32_t> num_gt_boxes;
    vector<int32_t> gt_classes;
    vector<float>   im_scale;
    vector<int32_t> gt_difficult;

    vector<void*>             buf_list;
    const vector<shape_type>& shapes = cfg.get_shape_type_list();
    ASSERT_EQ(10, shapes.size());
    size_t total_anchors = 34596;

    bbtargets.resize(shapes[0].get_element_count());
    bbtargets_mask.resize(shapes[1].get_element_count());
    labels_flat.resize(shapes[2].get_element_count());
    labels_mask.resize(shapes[3].get_element_count());
    im_shape.resize(shapes[4].get_element_count());
    gt_boxes.resize(shapes[5].get_element_count());
    num_gt_boxes.resize(shapes[6].get_element_count());
    gt_classes.resize(shapes[7].get_element_count());
    im_scale.resize(shapes[8].get_element_count());
    gt_difficult.resize(shapes[9].get_element_count());

    ASSERT_EQ(total_anchors * 4, bbtargets.size());
    ASSERT_EQ(total_anchors * 4, bbtargets_mask.size());
    ASSERT_EQ(total_anchors * 2, labels_flat.size());
    ASSERT_EQ(total_anchors * 2, labels_mask.size());
    ASSERT_EQ(2, im_shape.size());
    ASSERT_EQ(64 * 4, gt_boxes.size());
    ASSERT_EQ(1, num_gt_boxes.size());
    ASSERT_EQ(64, gt_classes.size());
    ASSERT_EQ(1, im_scale.size());
    ASSERT_EQ(64, gt_difficult.size());

    memset(bbtargets.data(), 0xFF, bbtargets.size() * sizeof(float));
    memset(bbtargets_mask.data(), 0xFF, bbtargets_mask.size() * sizeof(float));
    memset(labels_flat.data(), 0xFF, labels_flat.size() * sizeof(int32_t));
    memset(labels_mask.data(), 0xFF, labels_mask.size() * sizeof(int32_t));
    memset(im_shape.data(), 0xFF, im_shape.size() * sizeof(int32_t));
    memset(gt_boxes.data(), 0xFF, gt_boxes.size() * sizeof(float));
    memset(num_gt_boxes.data(), 0xFF, num_gt_boxes.size() * sizeof(int32_t));
    memset(gt_classes.data(), 0xFF, gt_classes.size() * sizeof(int32_t));
    memset(im_scale.data(), 0xFF, im_scale.size() * sizeof(float));
    memset(gt_difficult.data(), 0xFF, gt_difficult.size() * sizeof(int32_t));

    buf_list.push_back(bbtargets.data());
    buf_list.push_back(bbtargets_mask.data());
    buf_list.push_back(labels_flat.data());
    buf_list.push_back(labels_mask.data());
    buf_list.push_back(im_shape.data());
    buf_list.push_back(gt_boxes.data());
    buf_list.push_back(num_gt_boxes.data());
    buf_list.push_back(gt_classes.data());
    buf_list.push_back(im_scale.data());
    buf_list.push_back(gt_difficult.data());

    loader.load(buf_list, transformed_data);
}

TEST(localization_rcnn, compute_targets)
{
    // expected values generated via python localization example

    vector<box> gt_bb;
    vector<box> rp_bb;

    // ('gt_bb {0}', array([ 561.6,  329.6,  713.6,  593.6]))
    // ('rp_bb {1}', array([ 624.,  248.,  799.,  599.]))
    // xgt 638.1, rp 712.0, dx -0.419886363636
    // ygt 462.1, rp 424.0, dy  0.108238636364
    // wgt 153.0, rp 176.0, dw -0.140046073646
    // hgt 265.0, rp 352.0, dh -0.283901349612

    gt_bb.emplace_back(561.6, 329.6, 713.6, 593.6);
    rp_bb.emplace_back(624.0, 248.0, 799.0, 599.0);

    float dx_0_expected = -0.419886363636;
    float dy_0_expected = 0.108238636364;
    float dw_0_expected = -0.140046073646;
    float dh_0_expected = -0.283901349612;

    // ('gt_bb {0}', array([ 561.6,  329.6,  713.6,  593.6]))
    // ('rp_bb {1}', array([ 496.,  248.,  671.,  599.]))
    // xgt 638.1, rp 584.0, dx  0.307386363636
    // ygt 462.1, rp 424.0, dy  0.108238636364
    // wgt 153.0, rp 176.0, dw -0.140046073646
    // hgt 265.0, rp 352.0, dh -0.283901349612

    gt_bb.emplace_back(561.6, 329.6, 713.6, 593.6);
    rp_bb.emplace_back(496.0, 248.0, 671.0, 599.0);

    float dx_1_expected = 0.307386363636;
    float dy_1_expected = 0.108238636364;
    float dw_1_expected = -0.140046073646;
    float dh_1_expected = -0.283901349612;

    ASSERT_EQ(gt_bb.size(), rp_bb.size());

    vector<target> result = ::transformer::compute_targets(gt_bb, rp_bb);
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

TEST(localization_rcnn, provider)
{
    int   height               = 1000;
    int   width                = 1000;
    float fixed_scaling_factor = 1.6;

    nlohmann::json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    nlohmann::json js_local = {{"type", "localization_rcnn"},
                               {"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", {"bicycle", "person"}}};
    nlohmann::json augmentation = {{{"type", "image"},
                                    {"fixed_aspect_ratio", true},
                                    {"fixed_scaling_factor", fixed_scaling_factor},
                                    {"crop_enable", false},
                                    {"flip_enable", true}}};
    nlohmann::json js = {{"etl", {js_image, js_local}}, {"augmentation", augmentation}};

    shared_ptr<provider_interface> media   = provider_factory::create(js);
    auto                           oshapes = media->get_output_shapes();
    ASSERT_NE(nullptr, media);
    ASSERT_EQ(11, oshapes.size());

    string       target = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    vector<char> target_data;
    target_data.insert(target_data.begin(), target.begin(), target.end());
    // Image size is from the 006637.json target data file
    cv::Mat image(375, 500, CV_8UC3);
    image = cv::Scalar(50, 100, 200);
    vector<uint8_t> image_data;
    vector<char>    image_cdata;
    cv::imencode(".png", image, image_data);
    for (auto c : image_data)
    {
        image_cdata.push_back(c);
    };

    encoded_record_list in_buf;
    encoded_record      record;
    record.add_element(image_cdata);
    record.add_element(target_data);
    in_buf.add_record(record);

    fixed_buffer_map  out_buf(oshapes, 1);
    const shape_type& image_shape = media->get_output_shape("image");

    media->provide(0, in_buf, out_buf);

    int     output_width  = image_shape.get_shape()[0];
    int     output_height = image_shape.get_shape()[1];
    int     channels      = image_shape.get_shape()[2];
    cv::Mat result(output_height, output_width, CV_8UC(channels), out_buf["image"]->get_item(0));
    //cv::imwrite("localization_provider_source.png", image);
    //cv::imwrite("localization_provider.png", result);

    uint8_t* data = result.data;
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.cols; col++)
        {
            if (col < 800 && row < 600)
            {
                ASSERT_EQ(50, data[0]) << "row=" << row << ", col=" << col;
                ASSERT_EQ(100, data[1]) << "row=" << row << ", col=" << col;
                ASSERT_EQ(200, data[2]) << "row=" << row << ", col=" << col;
            }
            else
            {
                ASSERT_EQ(0, data[0]) << "row=" << row << ", col=" << col;
                ASSERT_EQ(0, data[1]) << "row=" << row << ", col=" << col;
                ASSERT_EQ(0, data[2]) << "row=" << row << ", col=" << col;
            }
            data += 3;
        }
    }
}

TEST(localization_rcnn, provider_channel_major)
{
    int   height               = 1000;
    int   width                = 1000;
    float fixed_scaling_factor = 1.6;

    nlohmann::json js_image = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", true}};
    nlohmann::json js_loc = {{"type", "localization_rcnn"},
                             {"height", height},
                             {"width", width},
                             {"max_gt_boxes", 64},
                             {"class_names", {"bicycle", "person"}}};
    nlohmann::json js_aug = {{{"type", "image"},
                              {"crop_enable", false},
                              {"fixed_aspect_ratio", true},
                              {"fixed_scaling_factor", fixed_scaling_factor},
                              {"flip_enable", true}}};
    nlohmann::json js = {{"etl", {js_image, js_loc}}, {"augmentation", js_aug}};

    auto media   = provider_factory::create(js);
    auto oshapes = media->get_output_shapes();
    ASSERT_NE(nullptr, media);
    ASSERT_EQ(11, oshapes.size());

    string       target = file_util::read_file_to_string(CURDIR "/test_data/006637.json");
    vector<char> target_data;
    target_data.insert(target_data.begin(), target.begin(), target.end());
    // Image size is from the 006637.json target data file
    cv::Mat image(375, 500, CV_8UC3);
    image = cv::Scalar(50, 100, 200);
    vector<uint8_t> image_data;
    vector<char>    image_cdata;
    cv::imencode(".png", image, image_data);
    for (auto c : image_data)
    {
        image_cdata.push_back(c);
    };

    encoded_record_list in_buf;
    encoded_record      record;
    record.add_element(image_cdata);
    record.add_element(target_data);
    in_buf.add_record(record);

    fixed_buffer_map  out_buf(oshapes, 1);
    const shape_type& image_shape = media->get_output_shape("image");

    media->provide(0, in_buf, out_buf);
    int     output_width  = image_shape.get_shape()[1];
    int     output_height = image_shape.get_shape()[2];
    cv::Mat result(output_height * 3, output_width, CV_8UC1, out_buf["image"]->get_item(0));
    cv::imwrite("localization_provider_channel_major.png", result);
    uint8_t* data = result.data;
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.cols; col++)
        {
            if (col < 800)
            {
                if (row >= 0 && row < 600)
                {
                    ASSERT_EQ(50, (int)*data) << "row=" << row << ", col=" << col;
                }
                else if (row >= 1000 && row < 1600)
                {
                    ASSERT_EQ(100, (int)*data) << "row=" << row << ", col=" << col;
                }
                else if (row >= 2000 && row < 2600)
                {
                    ASSERT_EQ(200, (int)*data) << "row=" << row << ", col=" << col;
                }
                else
                {
                    ASSERT_EQ(0, (int)*data) << "row=" << row << ", col=" << col;
                }
            }
            else
            {
                ASSERT_EQ(0, (int)*data) << "row=" << row << ", col=" << col;
            }
            data++;
        }
    }
}
