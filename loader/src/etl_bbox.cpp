#include <sstream>
#include "etl_bbox.hpp"

using namespace std;
using namespace nervana;
using namespace nlohmann;   // json stuff

ostream& operator<<(ostream& out, const nervana::bbox::box& b) {
    out << b.label << " (" << b.xmax << "," << b.xmin << ")(" << b.ymax << "," << b.ymin << ") "
        << b.difficult << " " << b.truncated;
    return out;
}

cv::Rect nervana::bbox::box::rect() const {
    return cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin);
}


nervana::bbox::decoded::decoded() {
}

nervana::bbox::extractor::extractor( const std::vector<std::string>& labels ) {
    for( int i=0; i<labels.size(); i++ ) {
        label_map.insert({labels[i],i});
    }
}

std::shared_ptr<nervana::bbox::decoded> nervana::bbox::extractor::extract(char* data, int size) {
    shared_ptr<decoded> rc = make_shared<decoded>();
    string buffer( data, size );
    json j = json::parse(buffer);
    if( j["object"].is_null() ) return rc;
    if( j["size"].is_null() ) return rc;
    auto object_list = j["object"];
    auto image_size = j["size"];
    if( image_size["width"].is_null() ) return rc;
    if( image_size["height"].is_null() ) return rc;
    if( image_size["depth"].is_null() ) return rc;
    rc->_height = image_size["height"];
    rc->_width = image_size["width"];
    rc->_depth = image_size["depth"];
    for( auto object : object_list ) {
        auto bndbox = object["bndbox"];
        box b;
        if( bndbox["xmax"].is_null() ) { rc = nullptr; return rc; }
        if( bndbox["xmin"].is_null() ) { rc = nullptr; return rc; }
        if( bndbox["ymax"].is_null() ) { rc = nullptr; return rc; }
        if( bndbox["ymin"].is_null() ) { rc = nullptr; return rc; }
        if( object["name"].is_null() ) { rc = nullptr; return rc; }
        b.xmax = bndbox["xmax"];
        b.xmin = bndbox["xmin"];
        b.ymax = bndbox["ymax"];
        b.ymin = bndbox["ymin"];
        if( !object["difficult"].is_null() ) b.difficult = object["difficult"];
        if( !object["truncated"].is_null() ) b.truncated = object["truncated"];
        string name = object["name"];
        auto found = label_map.find(name);
        if( found == label_map.end() ) {
            // did not find the label in the ctor supplied label list
            rc = nullptr;
            cout << "label '" << name << "' not found in label list" << endl;
            break;
        } else {
            b.label = found->second;
        }
        rc->_boxes.push_back(b);
    }
    return rc;
}

json nervana::bbox::extractor::create_box( const cv::Rect& rect, const string& label ) {
    json j = {{"bndbox",{{"xmax",rect.x+rect.width},{"xmin",rect.x},{"ymax",rect.y+rect.height},{"ymin",rect.y}}},{"name",label}};
    return j;
}

nlohmann::json nervana::bbox::extractor::create_metadata( const std::vector<nlohmann::json>& boxes ) {
    nlohmann::json j = nlohmann::json::object();
    j["object"] = boxes;
    j["size"] = {{"depth",3},{"height",256},{"width",256}};
    return j;
}

nervana::bbox::transformer::transformer() {

}

std::shared_ptr<bbox::decoded> nervana::bbox::transformer::transform(param_ptr _pptr, std::shared_ptr<bbox::decoded> boxes) {
    shared_ptr<image::params> pptr = static_pointer_cast<image::params>(_pptr);
    if( pptr->angle != 0 ) {
        return shared_ptr<bbox::decoded>();
    }
    shared_ptr<bbox::decoded> rc = make_shared<bbox::decoded>();
    cv::Rect crop = pptr->cropbox;
    for( box tmp : boxes->boxes() ) {
        box b = tmp;
        if( b.xmax <= crop.x ) {           // outside left
        } else if( b.xmin >= crop.x + crop.width ) {      // outside right
        } else if( b.ymax <= crop.y ) {   // outside above
        } else if( b.ymin >= crop.y + crop.height ) {     // outside below
        } else {
            if( b.xmin < crop.x ) {
                b.xmin = crop.x;
            }
            if( b.ymin < crop.y ) {
                b.ymin = crop.y;
            }
            if( b.xmax > crop.x + crop.width ) {
                b.xmax = crop.x + crop.width;
            }
            if( b.ymax > crop.y + crop.height ) {
                b.ymax = crop.y + crop.height;
            }
            rc->_boxes.push_back( b );
        }
        // cout << b.rect.x << ", " << b.rect.y << ", " << b.rect.width << ", " << b.rect.height << ", " << b.label << endl;
    }
    return rc;
}

void nervana::bbox::loader::load(char* data, int size, std::shared_ptr<bbox::decoded> boxes) {

}

