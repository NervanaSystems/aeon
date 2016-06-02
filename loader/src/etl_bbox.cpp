#include <sstream>
#include "etl_bbox.hpp"

using namespace std;
using namespace nervana;
using namespace nlohmann;   // json stuff

nervana::bbox::decoded::decoded( const char* data, int size ) {
    string buffer( data, size );
    json j = json::parse(buffer);
    for( auto bb : j["boxes"] ) {
        box new_box;
        int x = bb["x"];
        int y = bb["y"];
        int w = bb["w"];
        int h = bb["h"];
        int label = bb["label"];
        new_box.rect = cv::Rect( x, y, w, h );
        new_box.label = label;
        _boxes.push_back( new_box );
    }
}

nervana::bbox::extractor::extractor() {
}

media_ptr nervana::bbox::extractor::extract(char* data, int size) {
    return make_shared<decoded>(data,size);
}

json nervana::bbox::extractor::create_box( const cv::Rect& rect, int label ) {
    json j = {{"x",rect.x},{"y",rect.y},{"w",rect.width},{"h",rect.height},{"label",label}};
    return j;
}

media_ptr nervana::bbox::transformer::transform(settings_ptr settings, const media_ptr& media) {
    char* data;
    int size;
    return make_shared<decoded>(data,size);
}

void nervana::bbox::loader::load(char* data, int size, const media_ptr& media) {

}

