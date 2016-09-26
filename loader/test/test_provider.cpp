#include "gtest/gtest.h"
#include "gen_image.hpp"

#include "provider_factory.hpp"

#include "etl_image.hpp"
#include "etl_boundingbox.hpp"
#include "etl_label_map.hpp"
#include "json.hpp"
#include "cpio.hpp"
#include "util.hpp"
#include "provider_factory.hpp"
#include "loader.hpp"
#include "helpers.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

TEST(provider,empty_config)
{
    nlohmann::json js = {{"type","image,label"},
                         {"image", {
                            {"height",128},
                            {"width",128},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         };

    nervana::provider_factory::create(js);
}

TEST(provider,image)
{
    nlohmann::json js = {{"type","image,label"},
                         {"image", {
                            {"height",128},
                            {"width",128},
                            {"channel_major",false},
                            {"flip_enable",true}}},
                         {"label", {
                              {"binary",true}
                          }
                         }};

    auto media = nervana::provider_factory::create(js);
    const vector<nervana::shape_type>& oshapes = media->get_oshapes();

    size_t dsize = oshapes[0].get_byte_size();
    size_t tsize = 4;

    size_t batch_size = 128;

    buffer_out_array outBuf({dsize, tsize}, batch_size);

    auto files = image_dataset.GetFiles();
    ASSERT_NE(0,files.size());

    cpio::file_reader reader;
    EXPECT_EQ(reader.open(files[0]), true);
    buffer_in_array bp(2);
    buffer_in& data_p = *bp[0];
    buffer_in& target_p = *bp[1];
    for(int i=0; i<reader.itemCount()/2; i++) {
        reader.read(data_p);
        reader.read(target_p);
    }
    EXPECT_GT(data_p.get_item_count(),batch_size);
    for (int i=0; i<batch_size; i++ ) {
        media->provide(i, bp, outBuf);

//        cv::Mat mat(width,height,CV_8UC3,&dbuffer[0]);
//        string filename = "data" + to_string(i) + ".png";
//        cv::imwrite(filename,mat);
    }
    for (int i=0; i<batch_size; i++ ) {
        int target_value = unpack<int>(outBuf[1]->get_item(i));
        EXPECT_EQ(42+i,target_value);
    }
}

TEST(provider, argtype)
{

    {
        /* Create extractor with default num channels param */
        string cfgString = "{\"height\":10, \"width\":10}";
        auto js = nlohmann::json::parse(cfgString);
        image::config cfg{js};
        auto ic = make_shared<image::extractor>(cfg);
        EXPECT_EQ(ic->get_channel_count(), 3);
    }


    {
        string cfgString = R"(
            {
                "height": 30,
                "width" : 30,
                "angle" : [-20, 20],
                "scale" : [0.2, 0.8],
                "lighting" : [0.0, 0.1],
                "horizontal_distortion" : [0.75, 1.33],
                "flip_enable" : false
            }
        )";


        image::config itpj(nlohmann::json::parse(cfgString));

        // output the fixed parameters
        EXPECT_EQ(30,itpj.height);
        EXPECT_EQ(30,itpj.width);

        // output the random parameters
        default_random_engine r_eng(0);
        image::param_factory img_prm_maker(itpj);
        auto imgt = make_shared<image::transformer>(itpj);

        auto input_img_ptr = make_shared<image::decoded>(cv::Mat(256, 320, CV_8UC3));

        auto its = img_prm_maker.make_params(input_img_ptr);
    }
}

TEST(provider, blob)
{
    nlohmann::json js = {{"type","stereo_image,blob"},
                         {"image", {
                            {"height",360},
                            {"width",480},
                            {"channel_major",false}
                         }},
                         {"blob", {
                              {"output_type","float"},
                              {"output_count",480*360}
                          }
                         }};

    vector<char> input_left = read_file_contents(CURDIR"/test_data/img_2112_70.jpg");
    vector<char> input_right = read_file_contents(CURDIR"/test_data/img_2112_70.jpg");

    // flip input_left
    vector<uint8_t> tmp;
    auto mat = cv::imdecode(input_left,CV_LOAD_IMAGE_COLOR);
    cv::Size image_size = mat.size();
    cv::Mat flipped;
    cv::flip(mat, flipped, 1);
    cv::imwrite("left.jpg", flipped);
    cv::imencode(".jpg", flipped, tmp);
    input_left.clear();
    input_left.insert(input_left.begin(), tmp.begin(), tmp.end());

    // generate blob data, same size as image
    vector<float> target_data{480*360};
    iota(target_data.begin(), target_data.end(), 0);
    vector<char> target_cdata;
    char* p = (char*)target_data.data();
    for(int i=0; i<target_data.size()*sizeof(float); i++)
    {
        target_cdata.push_back(*p++);
    }

    // setup input and output buffers
    auto media = nervana::provider_factory::create(js);
    const vector<nervana::shape_type>& oshapes = media->get_oshapes();

    ASSERT_EQ(360,     int(js["image"]["height"]));
    ASSERT_EQ(480,     int(js["image"]["width"]));
    ASSERT_EQ(360*480, int(js["blob"]["output_count"]));
    ASSERT_EQ(3,oshapes.size());

    size_t left_size  = oshapes[0].get_byte_size();
    size_t right_size = oshapes[1].get_byte_size();
    size_t blob_size  = oshapes[2].get_byte_size();
    ASSERT_EQ(image_size.area()*3,left_size );
    ASSERT_EQ(image_size.area()*3,right_size );
    ASSERT_EQ(480*360*sizeof(float),blob_size );

    size_t batch_size = 1;

    buffer_out_array out_buf({left_size, right_size, blob_size}, batch_size);
    buffer_in_array  in_buf(3);
    in_buf[0]->add_item(input_left);
    in_buf[1]->add_item(input_right);
    in_buf[2]->add_item(target_cdata);

    // call the provider
    media->provide(0, in_buf, out_buf);

    cv::Mat output_left{image_size, CV_8UC3, out_buf[0]->data()};
    cv::imwrite("output_left.jpg", output_left);
    EXPECT_EQ(image_size, output_left.size());

    cv::Mat output_right{image_size, CV_8UC3, out_buf[1]->data()};
    cv::imwrite("output_right.jpg", output_right);
    EXPECT_EQ(image_size, output_right.size());

    float* fp = (float*)out_buf[2]->data();
    for(int i=0; i<target_data.size(); i++)
    {
        ASSERT_FLOAT_EQ(target_data[i], fp[i]);
    }
}
