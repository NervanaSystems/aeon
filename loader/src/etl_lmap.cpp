#include <sstream>
#include <iostream>
#include "etl_lmap.hpp"

using namespace std;
using namespace nervana;

lmap::config::config(nlohmann::json js) {
    parse_value(_labels, "labels", js, mode::REQUIRED);
    string type_string = "int32_t";
    parse_value(type_string, "type_string", js);

    otype = nervana::output_type(type_string);
    shape.push_back(otype.size);

    base_validate();
}

nervana::lmap::decoded::decoded() {
}

nervana::lmap::extractor::extractor( const lmap::config& cfg) {
    int index = 0;
    for( const string& label : cfg.labels() ) {
        _dictionary.insert({label,index++});
    }
}

shared_ptr<nervana::lmap::decoded> nervana::lmap::extractor::extract(const char* data, int size) {
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

nervana::lmap::transformer::transformer() {

}

shared_ptr<nervana::lmap::decoded> nervana::lmap::transformer::transform(
                                shared_ptr<nervana::lmap::params> pptr,
                                shared_ptr<nervana::lmap::decoded> media) {
    shared_ptr<lmap::decoded> rc = make_shared<lmap::decoded>();
    return rc;
}

void nervana::lmap::loader::load(char* data, shared_ptr<nervana::lmap::decoded> media) {

}

