#include <string>
#include <vector>
#include <istream>
#include <unordered_map>
#include <opencv2/core/core.hpp>

#include "etl_interface.hpp"

namespace nervana {
    namespace lmap {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class box;
    }
}

class nervana::lmap::decoded : public decoded_media {
    friend class transformer;
    friend class extractor;
public:
    decoded();
    virtual ~decoded() {}

    MediaType get_type() override { return MediaType::TARGET; }

    std::vector<int> get_data() const { return _labels; }

private:
    std::vector<int>    _labels;
};

class nervana::lmap::extractor : public extractor_interface {
public:
    extractor( const std::vector<std::string>& labels );
    extractor( std::istream& in );
    virtual ~extractor(){}
    virtual media_ptr extract(char*, int) override;

    std::unordered_map<std::string,int>  get_data() { return _dictionary; }

private:
    std::unordered_map<std::string,int>  _dictionary;
};

class nervana::lmap::transformer : public nervana::transformer_interface {
public:
    transformer();
    virtual ~transformer(){}
    virtual media_ptr transform(settings_ptr, const media_ptr&) override;
    virtual void fill_settings(settings_ptr) override;
private:
};

class nervana::lmap::loader : public nervana::loader_interface {
public:
    loader();
    virtual ~loader(){}
    virtual void load(char*, int, const media_ptr&) override;
private:
};
