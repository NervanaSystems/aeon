#include <string>
#include <unordered_map>

#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "json.hpp"
#include "box.hpp"

namespace nervana {
    namespace bbox {
        class decoded;
        class extractor;
        class transformer;
        class loader;
        class box;
    }
}

class nervana::bbox::box : public nervana::box {
public:
    bool difficult = false;
    bool truncated = false;
    int label;
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


class nervana::bbox::extractor : public nervana::interface::extractor<nervana::bbox::decoded> {
public:
    extractor( const std::vector<std::string>& label_list );
    virtual ~extractor(){}
    virtual std::shared_ptr<bbox::decoded> extract(const char*, int) override;
private:
     std::unordered_map<std::string,int> label_map;
};

class nervana::bbox::transformer : public nervana::interface::transformer<nervana::bbox::decoded, nervana::image::params> {
public:
    transformer();
    virtual ~transformer(){}
    virtual std::shared_ptr<bbox::decoded> transform(
                                            std::shared_ptr<image::params>,
                                            std::shared_ptr<bbox::decoded>) override;

private:
};

class nervana::bbox::loader : public nervana::interface::loader<nervana::bbox::decoded> {
public:
    loader();
    virtual ~loader(){}
    virtual void load(char*, std::shared_ptr<bbox::decoded>) override;

    void fill_info(count_size_type* cst) override
    {
        cst->count   = _load_count;
        cst->size    = _load_size;
        cst->type[0] = 'f';
    }

private:
    size_t _load_count;
    size_t _load_size;
};
