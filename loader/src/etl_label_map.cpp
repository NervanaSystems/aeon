#include <sstream>
#include <iostream>
#include "etl_label_map.hpp"

using namespace std;
using namespace nervana::label_map;

config::config(nlohmann::json js) {
    if(js.is_null()) {
        throw std::runtime_error("missing label_map config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config(config_list, js);

    // Derived types
    auto otype = nervana::output_type(type_string);
    add_shape_type({otype.size}, otype);

    if (type_string != "uint32_t") {
        throw std::runtime_error("Invalid load type for label map " + type_string);
    }
}

decoded::decoded() {
}

extractor::extractor( const label_map::config& cfg) {
    int index = 0;
    for( const string& label : cfg.labels ) {
        _dictionary.insert({label,index++});
    }
}

shared_ptr<decoded> extractor::extract(const char* data, int size) {
    auto rc = make_shared<decoded>();
    stringstream ss( string(data, size) );
    string label;
    while( ss >> label ) {
        auto l = _dictionary.find(label);
        if( l != _dictionary.end() ) {
            // found label
            rc->_labels.push_back(l->second);
        } else {
            // label not found in dictionary
            rc = nullptr;
            break;
        }
    }
    return rc;
}

transformer::transformer() {

}

shared_ptr<decoded> transformer::transform(shared_ptr<params> pptr, shared_ptr<decoded> media) {
    return media;
}

loader::loader(const nervana::label_map::config& cfg) :
    max_label_count{cfg.max_label_count()}
{
}

void loader::load(char* data, shared_ptr<decoded> media) {
    int i=0;
    uint32_t* data_p = (uint32_t*)data;
    for(; i<media->get_data().size() && i<max_label_count; i++ ) {
        data_p[i] = media->get_data()[i];
    }
    for(; i<max_label_count; i++ ) {
        data_p[i] = 0;
    }
}

