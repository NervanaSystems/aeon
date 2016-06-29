#pragma once

#include <vector>
#include <tuple>
#include <random>

#include "etl_interface.hpp"
#include "etl_bbox.hpp"
#include "params.hpp"
#include "util.hpp"
#include "box.hpp"

namespace nervana {

    namespace localization {
        class decoded;
        class params;
        class config;
        class target;
        class anchor;

        class extractor;
        class transformer;
        class loader;
    }

    class localization::target {
    public:
        target(float x, float y, float w, float h) :
            dx{x}, dy{y}, dw{w}, dh{h} {}
        float dx;
        float dy;
        float dw;
        float dh;
    };

    class localization::anchor {
    public:
        anchor(std::shared_ptr<const localization::config>);

        std::vector<box> inside_image_bounds(int width, int height);
    private:
        //    Generate anchor (reference) windows by enumerating aspect ratios X
        //    scales wrt a reference (0, 0, 15, 15) window.
        std::vector<box> generate_anchors();

        //    Enumerate a set of anchors for each aspect ratio wrt an anchor.
        std::vector<box> ratio_enum(const box& anchor, const std::vector<float>& ratios);

        //    Given a vector of widths (ws) and heights (hs) around a center
        //    (x_ctr, y_ctr), output a set of anchors (windows).
        std::vector<box> mkanchors(const std::vector<float>& ws, const std::vector<float>& hs, float x_ctr, float y_ctr);

        //    Enumerate a set of anchors for each scale wrt an anchor.
        std::vector<box> scale_enum(const box& anchor, const std::vector<float>& scales);

        //    Return width, height, x center, and y center for an anchor (window).
        std::tuple<float,float,float,float> whctrs(const box&);

        std::vector<box> add_anchors();

        std::shared_ptr<const localization::config> cfg;
        int conv_size;

        std::vector<box> all_anchors;
    };


//    class localization::params : public nervana::params {
//    public:

//        params() {}
//        void dump(std::ostream & = std::cout);

//        cv::Rect cropbox;
//        cv::Size2i output_size;
//        int angle = 0;
//        bool flip = false;
//        std::vector<float> lighting;  //pixelwise random values
//        float color_noise_std = 0;
//        std::vector<float> photometric;  // contrast, brightness, saturation
//        std::vector<std::string> label_list;
//    };

    class localization::config : public bbox::config {
    public:
        int images_per_batch = 1;
        int rois_per_image = 256;
        int min_size = 600;
        int max_size = 1000;
        int base_size = 16;
        float scaling_factor = 1.0 / 16.;
        std::vector<float> ratios = {0.5, 1, 2};
        std::vector<float> scales = {8, 16, 32};
        float negative_overlap = 0.3;  // negative anchors have < 0.3 overlap with any gt box
        float positive_overlap = 0.7;  // positive anchors have > 0.7 overlap with at least one gt box
        float foreground_fraction = 0.5;  // at most, positive anchors are 0.5 of the total rois

        bool set_config(nlohmann::json js) override;

    private:
        bool validate();
    };

    class localization::decoded : public bbox::decoded {
    public:
        decoded() {}
        virtual ~decoded() override {}

        // from transformer
        std::vector<int>    labels;
        std::vector<target> bbox_targets;
        std::vector<int>    anchor_index;
        std::vector<box>    anchors;

        float image_scale;
        cv::Size image_size;

    private:
    };


    class localization::extractor : public nervana::interface::extractor<localization::decoded> {
    public:
        extractor(std::shared_ptr<const localization::config>);

        virtual std::shared_ptr<localization::decoded> extract(const char* data, int size) override {
            auto rc = std::make_shared<localization::decoded>();
            auto bb = std::static_pointer_cast<bbox::decoded>(rc);
            bbox_extractor.extract(data, size, bb);
            return rc;
        }

        virtual ~extractor() {}
    private:
        bbox::extractor bbox_extractor;
    };

    class localization::transformer : public interface::transformer<localization::decoded, image::params> {
    public:
        transformer(std::shared_ptr<const localization::config> cfg);

        virtual ~transformer() {}

        std::shared_ptr<localization::decoded> transform(
                            std::shared_ptr<image::params> txs,
                            std::shared_ptr<localization::decoded> mp) override;

    private:
        std::tuple<float,cv::Size> calculate_scale_shape(cv::Size size);
        cv::Mat bbox_overlaps(const std::vector<box>& boxes, const std::vector<box>& query_boxes);
        std::vector<target> compute_targets(const std::vector<box>& gt_bb, const std::vector<box>& anchors);
        std::tuple<std::vector<int>,std::vector<target>,std::vector<int>> sample_anchors(const std::vector<int>& labels, const std::vector<target>& bbox_targets);

        std::shared_ptr<const localization::config> cfg;
        std::minstd_rand0 random;
        anchor  _anchor;
    };

    class localization::loader : public interface::loader<localization::decoded> {
    public:
        loader(std::shared_ptr<const localization::config> cfg) {}

        virtual ~loader() {}

        void load(char* buf, std::shared_ptr<localization::decoded> mp) override {

        }

        void fill_info(nervana::count_size_type*) override {

        }
    };
}
