#include <string>
#include "etl_interface.hpp"
#include "json.hpp"

namespace nervana {
    class decoded_bbox;
    class bbox_extractor;
    class bbox_transformer;
    class bbox_loader;
}

class nervana::decoded_bbox : public decoded_media {
public:
    decoded_bbox( const char* data, int size );
    virtual ~decoded_bbox() {}

    MediaType get_type() override { return MediaType::TARGET; }

private:
};


class nervana::bbox_extractor : public extractor_interface {
public:
    bbox_extractor();
    virtual ~bbox_extractor(){}
    virtual media_ptr extract(char*, int) override;
    static nlohmann::json create_box( int x, int y, int w, int h, int label );
private:
};

class nervana::bbox_transformer : public nervana::transformer_interface {
public:
    bbox_transformer();
    virtual ~bbox_transformer(){}
    virtual media_ptr transform(settings_ptr, const media_ptr&) override;
private:
};

class nervana::bbox_loader : public nervana::loader_interface {
public:
    bbox_loader();
    virtual ~bbox_loader(){}
    virtual void load(char*, int, const media_ptr&) override;
private:
};
