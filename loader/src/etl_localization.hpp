#pragma once

#include <vector>
#include <tuple>

#include "etl_interface.hpp"
#include "etl_bbox.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace localization {
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }

    class localization::decoded : public decoded_media {
    public:
        decoded(int index) :
            _index{index} {}
        virtual ~decoded() override {}

        inline MediaType get_type() override { return MediaType::TARGET; }
        inline int get_index() { return _index; }

    private:
        decoded() = delete;
        int _index;
    };


    class localization::extractor : public interface::extractor<localization::decoded> {
    public:
        extractor(std::shared_ptr<const json_config_parser> = nullptr){}

        virtual ~extractor() {}

        std::shared_ptr<localization::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw std::runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return std::make_shared<localization::decoded>(unpack_le<int>(buf));
        }
    };

    class localization::transformer : public interface::transformer<localization::decoded, nervana::params> {
    public:
        transformer(std::shared_ptr<const json_config_parser> = nullptr);

        virtual ~transformer() {}

        std::shared_ptr<localization::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<localization::decoded> mp) override { return mp; }

        //    Generate anchor (reference) windows by enumerating aspect ratios X
        //    scales wrt a reference (0, 0, 15, 15) window.
        static cv::Mat generate_anchors(int base_size, const std::vector<float>& ratios, const std::vector<float>& scales);
    private:
        //    Enumerate a set of anchors for each aspect ratio wrt an anchor.
        static cv::Mat ratio_enum(const std::vector<float>& anchor, const std::vector<float>& ratios);

        //    Given a vector of widths (ws) and heights (hs) around a center
        //    (x_ctr, y_ctr), output a set of anchors (windows).
        static cv::Mat mkanchors(const std::vector<float>& ws, const std::vector<float>& hs, float x_ctr, float y_ctr);

        //    Enumerate a set of anchors for each scale wrt an anchor.
        static cv::Mat scale_enum(const std::vector<float>& anchor, const std::vector<float>& scales);

        //    Return width, height, x center, and y center for an anchor (window).
        static std::tuple<float,float,float,float> whctrs(const std::vector<float>&);

        void add_anchors();

        int MAX_SIZE = 1000;
        int MIN_SIZE = 600;
        int ROI_PER_IMAGE = 256;  // number of anchors per image
        int IMG_PER_BATCH = 1;  // number of images per batch
        std::vector<std::string> CLASSES;  // list of CLASSES e.g. ['__background__', 'car', 'people',..]
        float SCALE = 1.0 / 16.;  // scaling factor of the image layers (e.g. VGG)

        // anchor variables
        std::vector<float> RATIOS = {0.5, 1, 2};  // aspect ratios to generate
        std::vector<float> SCALES = {128, 256, 512};  // box areas to generate

        float NEGATIVE_OVERLAP = 0.3;  // negative anchors have < 0.3 overlap with any gt box
        float POSITIVE_OVERLAP = 0.7;  // positive anchors have > 0.7 overlap with at least one gt box
        float FG_FRACTION = 0.5;  // at most, positive anchors are 0.5 of the total rois

        int conv_size = int(std::floor(MAX_SIZE * SCALE));
        float feat_stride = 1 / float(SCALE);
        int num_scales = SCALES.size() * RATIOS.size();
        int total_anchors = conv_size * conv_size * num_scales;
    };

    class localization::loader : public interface::loader<localization::decoded> {
    public:
        loader(std::shared_ptr<const json_config_parser> = nullptr) {}

        virtual ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<localization::decoded> mp) override
        {
            int index = mp->get_index();
            memcpy(buf, &index, bufSize);
        }
    };
}
