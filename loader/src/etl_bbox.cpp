#include <sstream>
#include "etl_bbox.hpp"

using namespace std;
using namespace nervana;

nervana::decoded_bbox::decoded_bbox( const char* data, int size ) {

}

nervana::bbox_extractor::bbox_extractor() {

}

media_ptr nervana::bbox_extractor::extract(char* data, int size) {
    return make_shared<decoded_bbox>(data,size);
}

nlohmann::json nervana::bbox_extractor::create_box( int x, int y, int w, int h, int label ) {
    // stringstream ss;
    // ss << "{{\"x\"," << x << "},{\"y\","<< y << "},{\"w\"," << w << "},{\"h\"," << h << "},{\"label\"," << label << "}}";
    // return ss.str();
    nlohmann::json j = {{"x",x},{"y",y},{"w",w},{"h",h},{"label",label}};
    return j;
}

media_ptr nervana::bbox_transformer::transform(settings_ptr settings, const media_ptr& media) {
    char* data;
    int size;
    return make_shared<decoded_bbox>(data,size);
}

void nervana::bbox_loader::load(char* data, int size, const media_ptr& media) {

}

