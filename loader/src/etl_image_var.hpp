#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_interface.hpp"
#include "params.hpp"


namespace nervana {
    namespace image_var {
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

    class image_var::params : public nervana::params {
        friend class image_var::param_factory;
    public:

        void dump(std::ostream & = std::cout);

        cv::Rect cropbox;
        bool flip = false;
        std::vector<float> lighting;  // pixelwise random values
        float color_noise_std = 0;
        std::vector<float> photometric;  // contrast, brightness, saturation
    private:
        params() {}
    };

    class image_var::param_factory : public interface::param_factory<image_var::decoded, image_var::params> {
    public:
        param_factory(std::shared_ptr<image_var::config> cfg,
                      std::default_random_engine& dre) : _cfg{cfg}, _dre{dre} {}
        ~param_factory() {}

        std::shared_ptr<image_var::params> make_params(std::shared_ptr<const decoded>);
    private:
        void scale_cropbox(const cv::Size2f&, cv::Rect&, float, float);

        std::shared_ptr<image_var::config> _cfg;
        std::default_random_engine& _dre;
    };

    class image_var::config : public json_config_parser {
    public:
        int min_size;
        int max_size;

        std::uniform_real_distribution<float> scale{1.0f, 1.0f};
        std::normal_distribution<float>       lighting{0.0f, 0.0f};
        std::uniform_real_distribution<float> photometric{0.0f, 0.0f};
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};
        std::bernoulli_distribution           flip{0};

        bool do_area_scale = false;
        bool channel_major = true;
        int channels = 3;

        bool set_config(nlohmann::json js) override
        {
            parse_req(min_size, "min_size", js);
            parse_req(max_size, "max_size", js);

            parse_opt(do_area_scale, "do_area_scale", js);
            parse_opt(channels, "channels", js);
            parse_opt(channel_major, "channel_major", js);

            auto dist_params = js["distribution"];
            parse_dist(scale, "scale", dist_params);
            parse_dist(lighting, "lighting", dist_params);
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

    class image_var::decoded : public decoded_media {
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

    class image_var::extractor : public interface::extractor<image_var::decoded> {
    public:
        extractor(std::shared_ptr<const image_var::config>);
        ~extractor() {}
        virtual std::shared_ptr<image_var::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
    private:
        int _pixel_type;
        int _color_mode;
    };

    class image_var::transformer : public interface::transformer<image_var::decoded, image_var::params> {
    public:
        transformer(std::shared_ptr<const image_var::config>);
        ~transformer() {}
        virtual std::shared_ptr<image_var::decoded> transform(
                                                std::shared_ptr<image_var::params>,
                                                std::shared_ptr<image_var::decoded>) override;

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

    class image_var::loader : public interface::loader<image_var::decoded> {
    public:
        loader(std::shared_ptr<const image_var::config>);
        ~loader() {}
        virtual void load(char*, std::shared_ptr<image_var::decoded>) override;

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
