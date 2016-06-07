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
    int xmax;
    int xmin;
    int ymax;
    int ymin;
    bool difficult;
    bool truncated;
    std::string name;
};

std::ostream& operator<<(std::ostream&,const nervana::bbox::box&);

class nervana::bbox::decoded : public decoded_media {
    friend class transformer;
    friend class extractor;
public:
    decoded();
    virtual ~decoded() {}

    MediaType get_type() override { return MediaType::TARGET; }

    std::vector<box> boxes() const { return _boxes; }
    int width() const { return _width; }
    int height() const { return _height; }
    int depth() const { return _depth; }

private:
    std::vector<box> _boxes;
    int _width;
    int _height;
    int _depth;
};


class nervana::bbox::extractor : public nervana::interface::extractor {
public:
    extractor();
    virtual ~extractor(){}
    virtual media_ptr extract(char*, int) override;
    static nlohmann::json create_box( const cv::Rect& rect, int label );
private:
};

class nervana::bbox::transformer : public nervana::interface::transformer {
public:
    transformer();
    virtual ~transformer(){}
    virtual media_ptr transform(settings_ptr, const media_ptr&) override;
    virtual void fill_settings(settings_ptr, const media_ptr&, std::default_random_engine &) override;

private:
};

class nervana::bbox::loader : public nervana::interface::loader {
public:
    loader();
    virtual ~loader(){}
    virtual void load(char*, int, const media_ptr&) override;
private:
};
