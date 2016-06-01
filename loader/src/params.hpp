#pragma once
#include <memory>
#include "argtype.hpp"

namespace nervana {
    class decoded_media;
    class image_extract_params;
    class image_transform_params;
    class image_load_params;
    class settings;
}

enum class MediaType {
    UNKNOWN = -1,
    IMAGE = 0,
    VIDEO = 1,
    AUDIO = 2,
    TEXT = 3,
    TARGET = 4,
};

/*  ABSTRACT INTERFACES */
class nervana::decoded_media {
public:
    virtual ~decoded_media() {}
    virtual MediaType get_type() = 0;
};

/*  ABSTRACT INTERFACES */
class nervana::settings {
public:
    virtual ~settings() {}
    virtual void set_
};

typedef std::shared_ptr<nervana::decoded_media>        media_ptr;
typedef std::shared_ptr<nervana::parameter_collection> param_ptr;
typedef std::shared_ptr<nervana::parameter_collection> settings_ptr;

class nervana::image_extract_params : public parameter_collection {
public:
    int channels;
    image_extract_params() {
        ADD_ARG(channels, "number of channels", "ch", "channels", 3, 1, 3);
    }
}

class nervana::image_transform_params : public parameter_collection {
public:
    float scale_pct;
    int angle;
    float cbs_range;
    float lighting_range;
    float aspect_ratio;
    bool area_scaling;
    bool flip;
    float crop_offset_range;

    image_transform_params() {
        // Required Params
        ADD_REQUIRED(height, "image height", "h", "height");
        ADD_REQUIRED(width, "image width", "w", "width");
        ADD_REQUIRED(scale_pct, "percentage of original image to scale crop", "s1", "scale_pct", 0, 100);

        // Optionals with some standard defaults
        ADD_OPTIONAL(channels, "number of channels", "ch", "channels", 3, 1, 3);
        ADD_OPTIONAL(angle, "rotation angle", "angle", "rotate-angle", 0, 0, 90);
        ADD_OPTIONAL(cbs_range, "augmentation range in pct to jitter contrast, brightness, saturation", "c2", "cbs_range", 0.0, 0.0, 1.0);
        ADD_OPTIONAL(lighting_range, "augmentation range in pct to jitter lighting", "c2", "lighting_range", 0.0, 0.0, 0.2);
        ADD_OPTIONAL(aspect_ratio, "aspect ratio to jitter", "a1", "aspect_ratio", 1.0, 1.0, 2.0);
        ADD_OPTIONAL(area_scaling, "whether to use area based scaling", "a2", "area_scaling", false);
        ADD_OPTIONAL(flip, "randomly flip?", "f1", "flip", false);
        ADD_OPTIONAL(center, "always center?", "c1", "center", true);
    }
}

class nervana::image_load_params : public parameter_collection {
public:
    int height;
    int width;
    bool chw;

    image_load_params() {
        // Required Params
        ADD_ARG(height, "image height", "h", "height");
        ADD_ARG(width, "image width", "w", "width");

        // Optional
        ADD_ARG(center, "always center?", "c1", "center", true);
    }
};
