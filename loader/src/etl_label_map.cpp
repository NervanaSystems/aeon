#include <sstream>
#include <iostream>
#include "etl_label_map.hpp"

using namespace std;
using namespace nervana;

label_map::config::config(nlohmann::json js) {
    parse_value(_labels, "labels", js, mode::REQUIRED);
    string type_string = "int32_t";
    parse_value(type_string, "type_string", js);

    otype = nervana::output_type(type_string);
    shape.push_back(otype.size);

    base_validate();
}

nervana::label_map::decoded::decoded() {
}

nervana::label_map::extractor::extractor( const label_map::config& cfg) {
    int index = 0;
    for( const string& label : cfg.labels() ) {
        _dictionary.insert({label,index++});
    }
}

shared_ptr<nervana::label_map::decoded> nervana::label_map::extractor::extract(const char* data, int size) {
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

nervana::label_map::transformer::transformer() {

}

shared_ptr<nervana::label_map::decoded> nervana::label_map::transformer::transform(
                                shared_ptr<nervana::label_map::params> pptr,
                                shared_ptr<nervana::label_map::decoded> media) {
    shared_ptr<label_map::decoded> rc = make_shared<label_map::decoded>();
    return rc;
}

void nervana::label_map::loader::load(char* data, shared_ptr<nervana::label_map::decoded> media) {

}

