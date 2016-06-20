#pragma once
#include <inttypes.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "params.hpp"


namespace nervana {
    namespace specgram {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }

    class specgram::params : public nervana::params {
    public:
        params() {}
        void dump(std::ostream & = std::cout) {}
    };

    class specgram::param_factory {
    public:
        param_factory(std::shared_ptr<specgram::config>);
        ~param_factory() {}

        std::shared_ptr<specgram::params> make_params(std::shared_ptr<const decoded>,
                                                   std::default_random_engine&);
    private:
        std::shared_ptr<specgram::config> _icp;
    };

    class specgram::config : public json_config_parser {
    public:

        // Required config values
        int sampling_freq;
        float input_duration_s;

        // Optional config values
        float frame_duration_s      = 10;
        float overlap_duration_s    = 3;
        std::string window_function = "hann";

        bool add_noise              = false;
        bool ctc_cost               = false;
        std::uniform_real_distribution<float> time_dilate{1.0f, 1.0f};

        // Derived config values
        uint32_t input_duration_n;
        uint32_t frame_duration_n;
        int stride_n;
        int time_steps; // width
        int freq_steps;  // height


        config(std::string argString)
        {
            auto js = nlohmann::json::parse(argString);

            parse_req(sampling_freq, "sampling_freq", js);
            parse_req(input_duration_s, "input_duration_s", js);

            parse_opt(frame_duration_s, "frame_duration_s", js);
            parse_opt(overlap_duration_s, "overlap_duration_s", js);
            parse_opt(window_function, "window_function", js);
            parse_opt(add_noise, "add_noise", js);
            parse_opt(ctc_cost, "ctc_cost", js);

            auto dist_params = js["distribution"];
            parse_dist(time_dilate, "time_dilate", dist_params);

            validate();

            // Figure out the derived values
            input_duration_n = input_duration_s * sampling_freq;
            frame_duration_n = frame_duration_s * sampling_freq;

            snap_to_nearest_pow2(frame_duration_n);

// This is already handled in snap_to_nearest_pow2
//            int prevpow = nextpow >> 1;
//            if ((nextpow - frame_duration_n) < (frame_duration_n - prevpow))
//                frame_duration_n = nextpow;
//            else
//                frame_duration_n = prevpow;

            int overlap_duration_n = overlap_duration_s * sampling_freq;

            stride_n = frame_duration_n - overlap_duration_n;
            time_steps = (input_duration_n - frame_duration_n) / stride_n + 1;
            freq_steps = frame_duration_n / 2 + 1;
        }

    private:
        bool validate()
        {
            bool isvalid = true;
            isvalid &= time_dilate.param().a() <= time_dilate.param().b();
            isvalid &= (0 < time_dilate.param().a()) && (time_dilate.param().b() < 2);
            return isvalid;
        }

        void snap_to_nearest_pow2(uint32_t &n)
        {
            uint32_t nextpow = n;
            nextpow--;
            nextpow |= nextpow >> 1;   // Divide by 2^k for consecutive doublings of k up to 32,
            nextpow |= nextpow >> 2;   // and then or the results.
            nextpow |= nextpow >> 4;
            nextpow |= nextpow >> 8;
            nextpow |= nextpow >> 16;
            nextpow++;
            uint32_t prevpow = nextpow >> 1;

            n = n > (nextpow - prevpow) / 2 ? nextpow : prevpow;
        }

    };

    class specgram::decoded : public image::decoded {
    public:
        decoded() {}
        decoded(cv::Mat img) : image::decoded(img) {}
        virtual ~decoded() override {}

//        virtual MediaType get_type() override { return MediaType::IMAGE; }
//        cv::Mat& get_specgram(int index) { return _images[index]; }
//        cv::Size2i get_specgram_size() const {return _images[0].size(); }
//        size_t get_specgram_count() const { return _images.size(); }

    private:
    };


    class specgram::extractor : public interface::extractor<specgram::decoded> {
    public:
        extractor(std::shared_ptr<const specgram::config>);
        ~extractor() {}
        virtual std::shared_ptr<specgram::decoded> extract(const char*, int) override;

    private:
        const float PI = 3.1415927;

        void none(int) {
        }

        void hann(int n) {
            for (int i = 0; i <= n; i++) {
                _window.at<float>(0, i) = 0.5 - 0.5 * cos((2.0 * PI * i) / n);
            }
        }

        void blackman(int n) {
            for (int i = 0; i <= n; i++) {
                _window.at<float>(0, i) = 0.42 -
                                           0.5 * cos((2.0 * PI * i) / n) +
                                           0.08 * cos(4.0 * PI * i / n);
            }
        }

        void hamming(int n) {
            for (int i = 0; i <= n; i++) {
                _window.at<float>(0, i) = 0.54 - 0.46 * cos((2.0 * PI * i) / n);
            }
        }

        void bartlett(int n) {
            for (int i = 0; i <= n; i++) {
                _window.at<float>(0, i) = 1.0 - 2.0 * fabs(i - n / 2.0) / n;
            }
        }

        cv::Mat _window;
    };


    class specgram::transformer : public interface::transformer<specgram::decoded, specgram::params> {
    public:
        transformer(std::shared_ptr<const specgram::config>) {}
        ~transformer() {}
        virtual std::shared_ptr<specgram::decoded> transform(
                                                std::shared_ptr<specgram::params>,
                                                std::shared_ptr<specgram::decoded>) override;

    private:
    };


    class specgram::loader : public interface::loader<specgram::decoded> {
    public:
        loader(std::shared_ptr<const specgram::config>);
        ~loader() {}
        virtual void load(char*, std::shared_ptr<specgram::decoded>) override;

    private:
        void split(cv::Mat&, char*);
        bool _channel_major;
    };
}
