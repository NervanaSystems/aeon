#include "gtest/gtest.h"
#include "gen_image.hpp"
#include "batchfile.hpp"

#include "provider_factory.hpp"

#include "etl_image.hpp"
#include "etl_label_test.hpp"
#include "etl_bbox.hpp"
#include "etl_lmap.hpp"
#include "json.hpp"
#include "batchfile.hpp"
#include "util.hpp"

extern gen_image image_dataset;

using namespace std;
using namespace nervana;

TEST(provider,image) {
    nlohmann::json js = {{"media","image"},
                         {"data_config",{{"height",128},{"width",128},{"channel_major",false},{"flip",true}}},
                         {"target_config",{}}};
    cout << js.dump(4) << endl;
    auto media = nervana::train_provider_factory::create(js);

    auto data_config = js["data_config"];
    int height = data_config["height"];
    int width = data_config["width"];
    size_t dsize = height * width * 3;
    size_t tsize = 4;

    vector<char> dbuffer(dsize);
    vector<char> tbuffer(tsize);

    auto files = image_dataset.GetFiles();
    ASSERT_NE(0,files.size());

    BatchFileReader reader(files[0]);
    for (int i=0; i<reader.itemCount()/2; i++ ) {
        Buffer data_p(0);
        reader.read(data_p);

        Buffer target_p(0);
        reader.read(target_p);

        BufferPair bp(&data_p, &target_p);
        media->provide_pair(0, &bp, &dbuffer[0], &tbuffer[0]);

        int target_value = unpack_le<int>(&tbuffer[0]);
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
        auto cfg = make_shared<image::config>();
        cfg->set_config(js);
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


        auto itpj = make_shared<image::config>();
        itpj->set_config(nlohmann::json::parse(cfgString));

        // output the fixed parameters
        EXPECT_EQ(30,itpj->height);
        EXPECT_EQ(30,itpj->width);

        // output the random parameters
        default_random_engine r_eng(0);
        image::param_factory img_prm_maker(itpj, r_eng);
        auto imgt = make_shared<image::transformer>(itpj);

        auto input_img_ptr = make_shared<image::decoded>(cv::Mat(256, 320, CV_8UC3));

        auto its = img_prm_maker.make_params(input_img_ptr);
        its->dump();
    }


    {
        string cfgString = R"(
            {
                "extract offset":  20,
                "dist_params/transform scale":  [-8, 8],
                "dist_params/transform shift":  [-5, 5]
            }
        )";
        auto lblcfg = make_shared<label_test::config>();
        lblcfg->set_config(nlohmann::json::parse(cfgString));

        auto dataFiles = image_dataset.GetFiles();
        ASSERT_GT(dataFiles.size(),0);
        string batchFileName = dataFiles[0];
        BatchFileReader bf(batchFileName);

        // Just get a single item
        Buffer data(0);
        Buffer labels(0);
        bf.read(data);
        bf.read(labels);

        bf.close();

        default_random_engine r_eng(0);
        default_random_engine r_eng_copy(0);
        auto prm_fcty = make_shared<label_test::param_factory>(lblcfg, r_eng);
        label_test::param_factory lbl_prm_maker(lblcfg, r_eng_copy);

        // Note we don't need the media to get params
        auto lstg = lbl_prm_maker.make_params(nullptr);

        int reference = ((int) *((int*)labels._data) + lblcfg->ex_offset)* lstg->scale + lstg->shift;

        // Take the int and do provision with it.
        auto lble = make_shared<label_test::extractor>(lblcfg);
        auto lblt = make_shared<label_test::transformer>(lblcfg);

        {
            auto lbll = make_shared<label_test::loader>(lblcfg);

            int reference_target = reference;
            int loaded_target = 0;
            provider<label_test::decoded, label_test::params> pp{lble, lblt, lbll, prm_fcty};
            auto ls2 = pp.provide(labels._data, 4, (char *)(&loaded_target), nullptr);
            EXPECT_EQ(ls2->scale, lstg->scale);
            EXPECT_EQ(ls2->shift, lstg->shift);
            EXPECT_EQ(reference_target, loaded_target);
        }

        {
            // This is a float loader that loads into a float with an offset
            string lArgString = R"(
                {
                    "load do float":  true,
                    "load offset": 0.8
                }
            )";
            auto flt_lbl_config = make_shared<label_test::config>();
            flt_lbl_config->set_config(nlohmann::json::parse(lArgString));

            auto lbll = make_shared<label_test::loader>(flt_lbl_config);

            float reference_target = reference + 0.8;
            float loaded_target = 0.0;
            provider<label_test::decoded, label_test::params> pp{lble, lblt, lbll};
            pp.provide(labels._data, 4, (char *)(&loaded_target), lstg);
            EXPECT_EQ(reference_target, loaded_target);
        }
    }
}

