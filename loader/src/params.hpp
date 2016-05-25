#pragma once
#include <memory>
#include "argtype.hpp"

namespace nervana {
    class decoded_media;
    class image_params;
    class settings;
}

enum class MediaType {
    UNKNOWN = -1,
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
};

/*  ABSTRACT INTERFACES */
class nervana::decoded_media {
public:
    virtual ~decoded_media();
    virtual MediaType get_type() = 0;

private:
    decoded_media();
};


typedef shared_ptr<nervana::decoded_media>        media_ptr_t;
typedef shared_ptr<nervana::ParameterCollection>  param_ptr_t;
typedef shared_ptr<nervana::settings>             settings_ptr_t;

class nervana::image_params : public ParameterCollection {
public:
    image_params() {
        // Required Params
        add<int>("height", "image height", "h", "height", true, 224, 1, 1024);
        add<int>("width", "image width", "w", "width", true, 224, 1, 1024);
        add<float>("scale_pct", "percentage of original image to scale crop", "s1", "scale_pct", true, 1.0, 0.0, 1.0);

        // Optionals with some standard defaults
        add<int>("channels", "number of channels", "ch", "channels", false, 3, 1, 3);
        add<int>("angle", "rotation angle", "angle", "rotate-angle", false, 0, 0, 90);
        add<float>("cbs_range", "augmentation range in pct to jitter contrast, brightness, saturation", "c2", "cbs_range", false, 0.0, 0.0, 1.0);
        add<float>("lighting_range", "augmentation range in pct to jitter lighting", "c2", "lighting_range", false, 0.0, 0.0, 0.2);
        add<float>("aspect_ratio", "aspect ratio to jitter", "a1", "aspect_ratio", false, 1.0, 1.0, 2.0);
        add<bool>("area_scaling", "whether to use area based scaling", "a2", "area_scaling", false, false);
        add<bool>("flip", "randomly flip?", "f1", "flip", false, false);
        add<bool>("center", "always center?", "c1", "center", false, true);
    }
};
