#pragma once

#include "provider_interface.hpp"
#include "etl_video.hpp"

namespace nervana {
    class video_only : public provider_interface {
    public:
        video_only(nlohmann::json js);
        void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);

    private:
        video::config               video_config;
        video::extractor            video_extractor;
        video::transformer          video_transformer;
        video::loader               video_loader;
        video::param_factory        video_factory;
    };
}
