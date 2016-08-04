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

nervana::bbox::config::config(nlohmann::json js)
{
    if(js.is_null()) {
        throw std::runtime_error("missing bbox config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config(config_list, js);

    // Derived values
    add_shape_type({max_bbox_count, 4*sizeof(float)}, type_string);
    label_map.clear();
    for( int i=0; i<labels.size(); i++ ) {
        label_map.insert({labels[i],i});
    }

    validate();
}

void nervana::bbox::config::validate() {
}

nervana::bbox::decoded::decoded() {
}

nervana::bbox::extractor::extractor(const std::unordered_map<std::string,int>& map) :
    label_map{map}
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

nervana::bbox::loader::loader(const bbox::config& cfg) :
    max_bbox{cfg.max_bbox_count}
{

}

void nervana::bbox::loader::load(const vector<void*>& outlist, shared_ptr<bbox::decoded> boxes) {
    float* data = (float*)outlist[0];
    size_t output_count = min(max_bbox, boxes->boxes().size());
    int i=0;
    for(; i<output_count; i++) {
        data[0] = boxes->boxes()[i].xmin;
        data[1] = boxes->boxes()[i].ymin;
        data[2] = boxes->boxes()[i].xmax;
        data[3] = boxes->boxes()[i].ymax;
        data += 4;
    }
    for(; i<max_bbox; i++) {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;
        data += 4;
    }
}

