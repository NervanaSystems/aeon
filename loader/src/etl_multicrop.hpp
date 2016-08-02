#include "etl_image.hpp"

namespace nervana {

    namespace multicrop {
        class config;
        class transformer;
    }


    class multicrop::config : public image::config {
    public:

        // Required config variables
        std::vector<float> multicrop_scales;

        // Optional config variables
        int crops_per_scale = 5;
        bool include_flips = true;

        // Derived config variables
        std::vector<cv::Point2f> offsets;
        cv::Size2i output_size;

        config(nlohmann::json js) :
            image::config::config(js)
        {
            // Parse required and optional variables
            parse_value(multicrop_scales, "multicrop_scales", js, mode::REQUIRED);
            parse_value(crops_per_scale, "crops_per_scale", js);
            parse_value(include_flips, "include_flips", js);

            if (!validate()) {
                throw std::runtime_error("invalid configuration values");
            }

            // Fill in derived variables
            offsets.push_back(cv::Point2f(0.5, 0.5)); // Center
            if (crops_per_scale == 5) {
                offsets.push_back(cv::Point2f(0.0, 0.0)); // NW
                offsets.push_back(cv::Point2f(0.0, 1.0)); // SW
                offsets.push_back(cv::Point2f(1.0, 0.0)); // NE
                offsets.push_back(cv::Point2f(1.0, 1.0)); // SE
            }
            output_size = cv::Size2i(height, width);

            // shape is going to be different because of multiple images
            uint32_t num_views = crops_per_scale * multicrop_scales.size() * (include_flips ? 2 : 1);
            shape.insert(shape.begin(), num_views);

            validate();
        }

    private:
        bool validate()
        {
            bool isvalid = true;
            isvalid &= ( crops_per_scale == 5 || crops_per_scale == 1);

            for (const float &s: multicrop_scales) {
                isvalid &= ( (0.0 < s) && (s < 1.0));
            }
            return isvalid;
        }
    };


    class multicrop::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const multicrop::config& cfg) : _cfg(cfg) {}
        ~transformer() {}
        virtual std::shared_ptr<image::decoded> transform(
                                                std::shared_ptr<image::params>,
                                                std::shared_ptr<image::decoded>) override;
    private:
        const multicrop::config& _cfg;

        void add_resized_crops(const cv::Mat&, std::shared_ptr<image::decoded>&, std::vector<cv::Rect>&);
    };
}