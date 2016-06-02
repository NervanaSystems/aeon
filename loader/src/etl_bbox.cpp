#include <sstream>
#include "etl_bbox.hpp"

using namespace std;
using namespace nervana;
using namespace nlohmann;   // json stuff

nervana::bbox::decoded::decoded() {
}

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

nervana::bbox::transformer::transformer() {

}

media_ptr nervana::bbox::transformer::transform(settings_ptr _settings, const media_ptr& media) {
    shared_ptr<image_settings> settings = static_pointer_cast<image_settings>(_settings);
    shared_ptr<bbox::decoded> boxes = static_pointer_cast<bbox::decoded>(media);
    shared_ptr<bbox::decoded> rc = make_shared<bbox::decoded>();
    cv::Rect crop = settings->cropbox;
    for( box tmp : boxes->get_data() ) {
        box b = tmp;
        if( b.rect.x + b.rect.width <= crop.x ) {           // outside left
        } else if( b.rect.x >= crop.x + crop.width ) {      // outside right
        } else if( b.rect.y + b.rect.height <= crop.y ) {   // outside above
        } else if( b.rect.y >= crop.y + crop.height ) {     // outside below
        } else {
            if( b.rect.x < crop.x ) {
                int dx = crop.x - b.rect.x;
                b.rect.x += dx;
                b.rect.width -= dx;
            }
            if( b.rect.y < crop.y ) {
                int dy = crop.y - b.rect.y;
                b.rect.y += dy;
                b.rect.height -= dy;
            }
            if( b.rect.x + b.rect.width > crop.x + crop.width ) {
                b.rect.width = crop.x + crop.width - b.rect.x;
            }
            if( b.rect.y + b.rect.height > crop.y + crop.height ) {
                b.rect.height = crop.y + crop.height - b.rect.y;
            }
            rc->_boxes.push_back( b );
        }
        // cout << b.rect.x << ", " << b.rect.y << ", " << b.rect.width << ", " << b.rect.height << ", " << b.label << endl;
    }
    return rc;
}

void nervana::bbox::transformer::fill_settings(settings_ptr) {

}

void nervana::bbox::loader::load(char* data, int size, const media_ptr& media) {

}

