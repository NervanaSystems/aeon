#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

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

    class image_var::config : public json_config_parser {
    public:
        int min_size;
        int max_size;

        std::bernoulli_distribution           flip{0};

        bool channel_major = true;
        int channels = 3;

        int32_t seed = 0; // Default is to seed deterministically

        config(nlohmann::json js)
        {
            parse_value(min_size, "min_size", js, mode::REQUIRED);
            parse_value(max_size, "max_size", js, mode::REQUIRED);

            parse_value(channels, "channels", js);
            parse_value(channel_major, "channel_major", js);
            parse_value(seed, "seed", js);

            auto dist_params = js["distribution"];
            parse_dist(flip, "flip", dist_params);
            validate();
        }

        virtual int num_crops() const { return 1; }

    private:
        config() = delete;
        bool validate() {
            return max_size >= min_size;
        }
    };

    class image_var::param_factory : public interface::param_factory<image_var::decoded, image_var::params> {
    public:
        param_factory(image_var::config& cfg) : _cfg{cfg}, _dre{0}
        {
            // A positive provided seed means to run deterministic with that seed
            if (_cfg.seed >= 0) {
                _dre.seed((uint32_t) _cfg.seed);
            } else {
                _dre.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }

        virtual ~param_factory() {}

        std::shared_ptr<image_var::params> make_params(std::shared_ptr<const decoded>);
    private:
        void scale_cropbox(const cv::Size2f&, cv::Rect&, float, float);

        image_var::config&         _cfg;
        std::default_random_engine _dre;
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image_var::decoded : public decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) : _image{img} {}
        virtual ~decoded() override {}

        cv::Mat& get_image() { return _image; }
        cv::Size2i get_image_size() const {return _image.size(); }
        int get_image_channels() const { return _image.channels(); }

    protected:
        cv::Mat _image;
    };

    class image_var::extractor : public interface::extractor<image_var::decoded> {
    public:
        extractor(const image_var::config&);
        virtual ~extractor() {}
        virtual std::shared_ptr<image_var::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
        void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
    private:
        int _pixel_type;
        int _color_mode;
    };

    class image_var::transformer : public interface::transformer<image_var::decoded, image_var::params> {
    public:
        transformer(const image_var::config&);
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
        loader(const image_var::config&);
        virtual ~loader() {}
        virtual void load(char*, std::shared_ptr<image_var::decoded>) override;

//        void fill_info(count_size_type* cst) override
//        {
//            cst->count   = _load_count;
//            cst->size    = _load_size;
//            cst->type[0] = 'u';
//        }

    private:
        size_t _load_size;
        void split(cv::Mat&, char*);
        bool _channel_major;
    };
}
