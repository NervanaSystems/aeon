#include "gtest/gtest.h"
#include "gen_image.hpp"

#include "provider_factory.hpp"

#include "etl_image.hpp"
#include "etl_bbox.hpp"
#include "etl_lmap.hpp"
#include "json.hpp"
#include "cpio.hpp"
#include "util.hpp"
#include "provider_factory.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

TEST(provider,image) {
    nlohmann::json js = {{"media","image_label"},
                         {"data_config", {{"type", "image"}, {"config", {{"height",128},{"width",128},{"channel_major",false},{"flip",true}}}}},
                         {"target_config", {{"type", "label"}, {"config", {}}}}};

    auto data_config = nervana::config_factory::create(js["data_config"]);
    size_t dsize = data_config->get_size_bytes();
    size_t tsize = 4;


    auto media = nervana::train_provider_factory::create(js);

//    vector<char> dbuffer(dsize);
//    vector<char> tbuffer(tsize);

    buffer_out dbuffer(1,dsize,1);
    buffer_out tbuffer(1,tsize,1);
    buffer_out_array outBuf({&dbuffer,&tbuffer});

    auto files = image_dataset.GetFiles();
    ASSERT_NE(0,files.size());

    CPIOFileReader reader;
    EXPECT_EQ(reader.open(files[0]), true);
    for (int i=0; i<reader.itemCount()/2; i++ ) {
        buffer_in data_p(0);
        reader.read(data_p);

        buffer_in target_p(0);
        reader.read(target_p);

        buffer_in_array bp{&data_p, &target_p};
        media->provide(0, bp, outBuf);

        int target_value = unpack_le<int>(tbuffer.data());
        EXPECT_EQ(42+i,target_value);
//        cv::Mat mat(width,height,CV_8UC3,&dbuffer[0]);
//        string filename = "data" + to_string(i) + ".png";
//        cv::imwrite(filename,mat);
    }
    cout << "cpio contains " << reader.itemCount() << endl;
}

TEST(provider, argtype) {

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
                "distribution":{
                    "angle" : [-20, 20],
                    "scale" : [0.2, 0.8],
                    "lighting" : [0.0, 0.1],
                    "aspect_ratio" : [0.75, 1.33],
                    "flip" : [false]
                }
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
