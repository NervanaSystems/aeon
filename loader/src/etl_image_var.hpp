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
    }

    class image_var::params : public nervana::params {
        friend class image_var::param_factory;
    public:

        void dump(std::ostream & = std::cout);

        bool flip = false;
        bool debug_deterministic = false;
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

        std::bernoulli_distribution           flip{0};

        bool channel_major = true;
        int channels = 3;

        bool set_config(nlohmann::json js) override
        {
            parse_req(min_size, "min_size", js);
            parse_req(max_size, "max_size", js);

            parse_opt(channels, "channels", js);
            parse_opt(channel_major, "channel_major", js);

            auto dist_params = js["distribution"];
            parse_dist(flip, "flip", dist_params);
            return validate();
        }

        virtual int num_crops() const { return 1; }

    private:
        bool validate() {
            return max_size >= min_size;
        }
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image_var::decoded : public decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) : _image{img} {}
        virtual ~decoded() override {}

        virtual MediaType get_type() override { return MediaType::IMAGE; }
        cv::Mat& get_image() { return _image; }
        cv::Size2i get_image_size() const {return _image.size(); }
        int get_image_channels() const { return _image.channels(); }

    protected:
        cv::Mat _image;
    };

    class image_var::extractor : public interface::extractor<image_var::decoded> {
    public:
        extractor(std::shared_ptr<const image_var::config>);
        ~extractor() {}
        virtual std::shared_ptr<image_var::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
        void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
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
        int min_size;
        int max_size;
    };

    class image_var::loader : public interface::loader<image_var::decoded> {
    public:
        loader(std::shared_ptr<const image_var::config>);
        ~loader() {}
        virtual void load(char*, std::shared_ptr<image_var::decoded>) override;

//        void fill_info(count_size_type* cst) override
//        {
//            cst->count   = _load_count;
//            cst->size    = _load_size;
//            cst->type[0] = 'u';
//        }

    private:
        size_t _load_count;
        size_t _load_size;
        void split(cv::Mat&, char*);
        bool _channel_major;
    };
}
