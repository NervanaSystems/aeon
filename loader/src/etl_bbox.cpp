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

nervana::bbox::config::config(nlohmann::json js, bool ignore_errors)
{
    label_map.clear();
    vector<string> label_list;
    parse_value(label_list, "labels", js, mode::REQUIRED);
    for( int i=0; i<label_list.size(); i++ ) {
        label_map.insert({label_list[i],i});
    }
    // TODO -- need to fill in the shape and type information here
    string type_string = "float";
    parse_value(type_string,    "type_string", js);
    parse_value(height,         "height", js, mode::REQUIRED);
    parse_value(width,          "width", js, mode::REQUIRED);
    parse_value(max_bbox,       "max_bbox_count", js, mode::REQUIRED);

    shape = {uint32_t(max_bbox), uint32_t(4*sizeof(float))};
    otype = nervana::output_type(type_string);

    if(!ignore_errors) validate();
}

void nervana::bbox::config::validate() {
    base_validate();
}


nervana::bbox::decoded::decoded() {
}

nervana::bbox::extractor::extractor(const config& cfg) :
    label_map{cfg.label_map}
{
}

void nervana::bbox::extractor::extract(const char* data, int size, std::shared_ptr<bbox::decoded>& rc) {
    string buffer( data, size );
    json j = json::parse(buffer);
    if( j["object"].is_null() ) { rc = nullptr; return; }
    if( j["size"].is_null() ) { rc = nullptr; return; }
    auto object_list = j["object"];
    auto image_size = j["size"];
    if( image_size["width"].is_null() ) { rc = nullptr; return; }
    if( image_size["height"].is_null() ) { rc = nullptr; return; }
    if( image_size["depth"].is_null() ) { rc = nullptr; return; }
    rc->_height = image_size["height"];
    rc->_width = image_size["width"];
    rc->_depth = image_size["depth"];
    for( auto object : object_list ) {
        auto bndbox = object["bndbox"];
        box b;
        if( bndbox["xmax"].is_null() ) { rc = nullptr; return; }
        if( bndbox["xmin"].is_null() ) { rc = nullptr; return; }
        if( bndbox["ymax"].is_null() ) { rc = nullptr; return; }
        if( bndbox["ymin"].is_null() ) { rc = nullptr; return; }
        if( object["name"].is_null() ) { rc = nullptr; return; }
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
}

shared_ptr<nervana::bbox::decoded> nervana::bbox::extractor::extract(const char* data, int size) {
    shared_ptr<decoded> rc = make_shared<decoded>();
    extract(data, size, rc);
    return rc;
}

nervana::bbox::transformer::transformer(const bbox::config&) {}

shared_ptr<bbox::decoded> nervana::bbox::transformer::transform(shared_ptr<image::params> pptr, shared_ptr<bbox::decoded> boxes) {
    if( pptr->angle != 0 ) {
        return shared_ptr<bbox::decoded>();
    }
    shared_ptr<bbox::decoded> rc = make_shared<bbox::decoded>();
    cv::Rect crop = pptr->cropbox;
    float x_scale = (float)(pptr->output_size.width)  / (float)(boxes->width());
    float y_scale = (float)(pptr->output_size.height) / (float)(boxes->height());
    for( box tmp : boxes->boxes() ) {
        box b = tmp;
        if( b.xmax <= crop.x ) {                      // outside left
        } else if( b.xmin >= crop.x + crop.width ) {  // outside right
        } else if( b.ymax <= crop.y ) {               // outside above
        } else if( b.ymin >= crop.y + crop.height ) { // outside below
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

            // now rescale box
            b.xmin = (decltype(b.xmin))round((float)b.xmin * x_scale);
            b.xmax = (decltype(b.xmax))round((float)b.xmax * x_scale);
            b.ymin = (decltype(b.ymin))round((float)b.ymin * y_scale);
            b.ymax = (decltype(b.ymax))round((float)b.ymax * y_scale);

            rc->_boxes.push_back( b );
        }
        // cout << b.rect << ", " << b.label << endl;
    }
    return rc;
}

nervana::bbox::loader::loader(const bbox::config&) {

}

void nervana::bbox::loader::load(char* data, shared_ptr<bbox::decoded> boxes) {

}

