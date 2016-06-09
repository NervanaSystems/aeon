#include <string>
#include <unordered_map>
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
    int xmax = 0;
    int xmin = 0;
    int ymax = 0;
    int ymin = 0;
    bool difficult = false;
    bool truncated = false;
    int label;

    cv::Rect rect() const;
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
    virtual std::shared_ptr<bbox::decoded> extract(char*, int) override;
    static nlohmann::json create_box( const cv::Rect& rect, const std::string& label );
    static nlohmann::json create_metadata( const std::vector<nlohmann::json>& boxes );
private:
     std::unordered_map<std::string,int> label_map;
};

class nervana::bbox::transformer : public nervana::interface::transformer<nervana::bbox::decoded> {
public:
    transformer();
    virtual ~transformer(){}
    virtual std::shared_ptr<bbox::decoded> transform(settings_ptr, std::shared_ptr<bbox::decoded>) override;
    virtual void fill_settings(settings_ptr, std::shared_ptr<bbox::decoded>, std::default_random_engine &) override;

private:
};

class nervana::bbox::loader : public nervana::interface::loader<nervana::bbox::decoded> {
public:
    loader();
    virtual ~loader(){}
    virtual void load(char*, int, std::shared_ptr<bbox::decoded>) override;
private:
};
