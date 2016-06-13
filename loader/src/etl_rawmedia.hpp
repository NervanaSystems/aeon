#pragma once

#include "etl_interface.hpp"
#include "params.hpp"
#include <libavcodec/avcodec.h>
#include <libavutil/common.h>


namespace nervana {
    namespace rawmedia {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }

    class rawmedia::params : public nervana::params {
    public:
        params() {}
        void dump(std::ostream & = std::cout) {}
    };

    class rawmedia::param_factory {
    public:
        param_factory(std::shared_ptr<rawmedia::config>);
        ~param_factory() {}

        std::shared_ptr<rawmedia::params> make_params(std::shared_ptr<const decoded>,
                                                      std::default_random_engine&);
    private:
        std::shared_ptr<rawmedia::config> _icp;
    };

    class rawmedia::config : public json_config_parser {
    public:

        string media_type;
        AVMediaType av_media_type = AVMEDIA_TYPE_UNKOWN;

        config(std::string argString)
        {
            auto js = nlohmann::json::parse(argString);

            parse_req(media_type, "media_type", js);

            auto avt = _avtype_map.find(media_type);
            if (avt == _avtype_map.end()) {
                throw runtime_error("Unsupported media");
            } else {
                av_media_type = avt->second;
            }

            validate();
        }

    private:
        bool validate() {}

        std::unordered_map<std::string, AVMediaType> _avtype_map = {{"audio", AVMEDIA_TYPE_AUDIO},
                                                                    {"video", AVMEDIA_TYPE_VIDEO}};

    };

    class rawmedia::decoded : public decoded_media {
    public:
        decoded(MediaType m) : _media_type(m) {}

        virtual ~decoded() override
        {
            for (uint i = 0; i < _bufs.size(); i++) {
                delete[] _bufs[i];
            }
        }

        virtual MediaType get_type() override { return _media_type; }

        inline void reset() { _data_size = 0; }
        inline void set_sample_size(int sample_size) { _sample_size = sample_size; }
        inline char* get_buf(int idx) { return _bufs[idx]; }
        inline int get_size() { return _bufs.size(); }
        inline int get_buf_size() { return _buf_size; }
        inline int get_data_size() { return _data_size; }
        inline int get_sample_size() { return _sample_size; }

        void add_bufs(int count, int size);
        void fill_bufs(char** frames, int frame_size);
        void grow_bufs(int grow);
        void copy_data(char* buf, int buf_size);

    private:
        int _buf_size    = 0;
        int _data_size   = 0;
        int _sample_size = 0;
        std::vector<char*> _bufs;
        MediaType _media_type;
    };


    class rawmedia::extractor : public interface::extractor<rawmedia::decoded> {
    public:
        extractor(std::shared_ptr<const rawmedia::config> cfg)
        {
            _av_media_type = cfg->av_media_type;
        }

        ~extractor() {}
        virtual std::shared_ptr<rawmedia::decoded> extract(const char*, int) override;

    private:
        void decode_frame(AVPacket* packet,
                          std::shared_ptr<rawmedia::decoded> raw,
                          int stream,
                          int itemSize);

        AVMediaType      _av_media_type;
        AVFormatContext* _format;
        AVCodecContext*  _codec;
    };

}
