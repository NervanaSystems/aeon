#pragma once

#include <vector>
#include <tuple>

#include "etl_interface.hpp"
#include "etl_bbox.hpp"
#include "params.hpp"
#include "util.hpp"
#include "box.hpp"

namespace nervana {

    namespace localization {
        class decoded;

        class extractor;
        class transformer;
        class loader;
        class anchor;
    }

    class localization::anchor {
    public:
        anchor(int max_size, int min_size);

        std::vector<box> inside_im_bounds(int width, int height);

        std::vector<box> add_anchors();

        //    Generate anchor (reference) windows by enumerating aspect ratios X
        //    scales wrt a reference (0, 0, 15, 15) window.
        std::vector<box> generate_anchors(int base_size, const std::vector<float>& ratios, const std::vector<float>& scales);
    private:
        //    Enumerate a set of anchors for each aspect ratio wrt an anchor.
        std::vector<box> ratio_enum(const box& anchor, const std::vector<float>& ratios);

        //    Given a vector of widths (ws) and heights (hs) around a center
        //    (x_ctr, y_ctr), output a set of anchors (windows).
        std::vector<box> mkanchors(const std::vector<float>& ws, const std::vector<float>& hs, float x_ctr, float y_ctr);

        //    Enumerate a set of anchors for each scale wrt an anchor.
        std::vector<box> scale_enum(const box& anchor, const std::vector<float>& scales);

        //    Return width, height, x center, and y center for an anchor (window).
        std::tuple<float,float,float,float> whctrs(const box&);


        int MAX_SIZE;
        int MIN_SIZE;
//        int ROI_PER_IMAGE = 256;  // number of anchors per image
//        int IMG_PER_BATCH = 1;  // number of images per batch
        std::vector<std::string> CLASSES;  // list of CLASSES e.g. ['__background__', 'car', 'people',..]
        float SCALE = 1.0 / 16.;  // scaling factor of the image layers (e.g. VGG)

        // anchor variables
        std::vector<float> RATIOS = {0.5, 1, 2};  // aspect ratios to generate
        std::vector<float> SCALES = {128, 256, 512};  // box areas to generate

        int conv_size;
//        float feat_stride = 1 / float(SCALE);

        std::vector<box> all_anchors;
    };

    class localization::decoded : public bbox::decoded {
    public:
        decoded(int index) {}
        virtual ~decoded() override {}

    private:
        decoded() = delete;
    };


    class localization::extractor : public nervana::bbox::extractor {
    public:
        extractor( const std::vector<std::string>& label_list ) :
            bbox::extractor::extractor{label_list}
        {
        }

        virtual ~extractor() {}
    };

    class localization::transformer : public interface::transformer<localization::decoded, nervana::params> {
    public:
        transformer(std::shared_ptr<const json_config_parser> = nullptr);

        virtual ~transformer() {}

        std::shared_ptr<localization::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<localization::decoded> mp) override;

    private:
        std::tuple<float,cv::Size> calculate_scale_shape(cv::Size size);
        cv::Mat bbox_overlaps(const std::vector<box>& boxes, const std::vector<box>& query_boxes);
        void compute_targets(const std::vector<box>& gt_bb, const std::vector<box>& anchors);

        int MAX_SIZE = 1000;
        int MIN_SIZE = 600;

        float NEGATIVE_OVERLAP = 0.3;  // negative anchors have < 0.3 overlap with any gt box
        float POSITIVE_OVERLAP = 0.7;  // positive anchors have > 0.7 overlap with at least one gt box
        float FG_FRACTION = 0.5;  // at most, positive anchors are 0.5 of the total rois

        anchor  _anchor;
    };

    class localization::loader : public interface::loader<localization::decoded> {
    public:
        loader(std::shared_ptr<const json_config_parser> = nullptr) {}

        virtual ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<localization::decoded> mp) override
        {
        }
    };
}
