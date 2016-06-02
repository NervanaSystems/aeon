#include <string>
#include <opencv2/core/core.hpp>

#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "json.hpp"

namespace nervana {
    namespace bbox {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class box;
    }
}

class nervana::bbox::box {
public:
    cv::Rect    rect;
    int         label;
};

class nervana::bbox::decoded : public decoded_media {
    friend class transformer;
public:
    decoded();
    decoded( const char* data, int size );
    virtual ~decoded() {}

    MediaType get_type() override { return MediaType::TARGET; }

    std::vector<box> get_data() { return _boxes; }

private:
    std::vector<box> _boxes;
};


class nervana::bbox::extractor : public extractor_interface {
public:
    extractor();
    virtual ~extractor(){}
    virtual media_ptr extract(char*, int) override;
    static nlohmann::json create_box( const cv::Rect& rect, int label );
private:
};

class nervana::bbox::transformer : public nervana::transformer_interface {
public:
    transformer();
    virtual ~transformer(){}
    virtual media_ptr transform(settings_ptr, const media_ptr&) override;
    virtual void fill_settings(settings_ptr) override;
private:
};

class nervana::bbox::loader : public nervana::loader_interface {
public:
    loader();
    virtual ~loader(){}
    virtual void load(char*, int, const media_ptr&) override;
private:
};
