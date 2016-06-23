#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_interface.hpp"
#include "params.hpp"


namespace nervana {
    namespace image {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;

        // These functions may be common across different transformers
        void resize(const cv::Mat&, cv::Mat&, const cv::Size2i& );
        void shift_cropbox(const cv::Size2f &, cv::Rect &, float, float);
    }
    namespace video {
        class config;     // Forward decl for friending
    }

    namespace multicrop {
        class config;
        class transformer;
    }

    class image::params : public nervana::params {
        friend class image::param_factory;
        params() {}
    public:

        void dump(std::ostream & = std::cout);

        cv::Rect cropbox;
        cv::Size2i output_size;
        int angle = 0;
        bool flip = false;
        std::vector<float> lighting;  // pixelwise random values
        float color_noise_std = 0;
        std::vector<float> photometric;  // contrast, brightness, saturation
    };

    class image::param_factory : public interface::param_factory<image::decoded, image::params> {
    public:
        param_factory(std::shared_ptr<image::config> cfg,
                      std::default_random_engine& dre) : _cfg{cfg}, _dre{dre} {}
        ~param_factory() {}

        std::shared_ptr<image::params> make_params(std::shared_ptr<const decoded>);
    private:
        void scale_cropbox(const cv::Size2f&, cv::Rect&, float, float);

        std::shared_ptr<image::config> _cfg;
        std::default_random_engine& _dre;
    };

    class image::config : public json_config_parser {
        friend class video::config;
    public:
        int height;
        int width;

        std::uniform_real_distribution<float> scale{1.0f, 1.0f};
        std::uniform_int_distribution<int>    angle{0, 0};
        std::normal_distribution<float>       lighting{0.0f, 0.0f};
        std::uniform_real_distribution<float> aspect_ratio{1.0f, 1.0f};
        std::uniform_real_distribution<float> photometric{0.0f, 0.0f};
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};
        std::bernoulli_distribution           flip{0};

        bool do_area_scale = false;
        bool channel_major = true;
        int channels = 3;

        bool set_config(nlohmann::json js) override
        {
            parse_req(height, "height", js);
            parse_req(width, "width", js);

            parse_opt(do_area_scale, "do_area_scale", js);
            parse_opt(channels, "channels", js);
            parse_opt(channel_major, "channel_major", js);

            auto dist_params = js["distribution"];
            parse_dist(angle, "angle", dist_params);
            parse_dist(scale, "scale", dist_params);
            parse_dist(lighting, "lighting", dist_params);
            parse_dist(aspect_ratio, "aspect_ratio", dist_params);
            parse_dist(photometric, "photometric", dist_params);
            parse_dist(crop_offset, "crop_offset", dist_params);
            parse_dist(flip, "flip", dist_params);
            return validate();
        }

        virtual int num_crops() const { return 1; }

    private:
        bool validate() {
            return crop_offset.param().a() <= crop_offset.param().b();
        }
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image::decoded : public decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) { _images.push_back(img); }
        bool add(cv::Mat img) {
            _images.push_back(img);
            return all_images_are_same_size();
        }
        bool add(const std::vector<cv::Mat>& images) {
            for( auto mat : images ) {
                _images.push_back(mat);
            }
            return all_images_are_same_size();
        }
        virtual ~decoded() override {}

        virtual MediaType get_type() override { return MediaType::IMAGE; }
        cv::Mat& get_image(int index) { return _images[index]; }
        cv::Size2i get_image_size() const {return _images[0].size(); }
        int get_image_channels() const { return _images[0].channels(); }
        size_t get_image_count() const { return _images.size(); }
        size_t get_size() const {
            return get_image_size().area() * get_image_channels() * get_image_count();
        }

    protected:
        bool all_images_are_same_size() {
            for( int i=1; i<_images.size(); i++ ) {
                if(_images[0].size()!=_images[i].size()) return false;
            }
            return true;
        }
        std::vector<cv::Mat> _images;
    };



    class image::extractor : public interface::extractor<image::decoded> {
    public:
        extractor(std::shared_ptr<const image::config>);
        ~extractor() {}
        virtual std::shared_ptr<image::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
    private:
        int _pixel_type;
        int _color_mode;
    };


    class image::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(std::shared_ptr<const image::config>);
        ~transformer() {}
        virtual std::shared_ptr<image::decoded> transform(
                                                std::shared_ptr<image::params>,
                                                std::shared_ptr<image::decoded>) override;

    private:
        void rotate(const cv::Mat& input, cv::Mat& output, int angle);
        void lighting(cv::Mat& inout, std::vector<float>, float color_noise_std);
        void cbsjitter(cv::Mat& inout, const std::vector<float>&);

        // These are the eigenvectors of the pixelwise covariance matrix
        const float _CPCA[3][3];
        const cv::Mat CPCA;

        // These are the square roots of the eigenvalues of the pixelwise covariance matrix
        const cv::Mat CSTD;

        // This is the set of coefficients for converting BGR to grayscale
        const cv::Mat GSCL;
    };


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

        bool set_config(nlohmann::json js) override
        {
            image::config::set_config(js);
            // Parse required and optional variables
            parse_req(multicrop_scales, "multicrop_scales", js);
            parse_opt(crops_per_scale, "crops_per_scale", js);
            parse_opt(include_flips, "include_flips", js);

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
            return validate();
        }

        int num_crops() const override { return static_cast<int>(multicrop_scales.size()); }

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
        transformer(std::shared_ptr<multicrop::config> cfg) : _cfg(cfg) {}
        ~transformer() {}
        virtual std::shared_ptr<image::decoded> transform(
                                                std::shared_ptr<image::params>,
                                                std::shared_ptr<image::decoded>) override;
    private:
        std::shared_ptr<multicrop::config> _cfg;

        void add_resized_crops(const cv::Mat&, std::shared_ptr<image::decoded>&, std::vector<cv::Rect>&);
    };


    class image::loader : public interface::loader<image::decoded> {
    public:
        loader(std::shared_ptr<const image::config>);
        ~loader() {}
        virtual void load(char*, std::shared_ptr<image::decoded>) override;

        void fill_info(count_size_type* cst) override
        {
            cst->count   = _load_count;
            cst->size    = _load_size;
            cst->type[0] = 'u';
        }

    private:
        size_t _load_count;
        size_t _load_size;
        void split(cv::Mat&, char*);
        bool _channel_major;
    };
}
