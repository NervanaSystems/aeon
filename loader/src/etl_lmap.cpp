#include <sstream>
#include <iostream>
#include "etl_lmap.hpp"

using namespace std;
using namespace nervana;

nervana::lmap::decoded::decoded() {
}

nervana::lmap::extractor::extractor( const vector<string>& labels ) {
    int index = 0;
    for( const string& label : labels ) {
        _dictionary.insert({label,index++});
    }
}

nervana::lmap::extractor::extractor( std::istream& in ) {
    int index = 0;
    string label;
    while( in >> label ) {
        _dictionary.insert({label,index++});
    }
}

std::shared_ptr<nervana::lmap::decoded> nervana::lmap::extractor::extract(char* data, int size) {
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

std::shared_ptr<nervana::lmap::decoded> nervana::lmap::transformer::transform(param_ptr _pptr, std::shared_ptr<nervana::lmap::decoded> media) {
    shared_ptr<lmap::decoded> rc = make_shared<lmap::decoded>();
    return rc;
}

void nervana::lmap::loader::load(char* data, int size, std::shared_ptr<nervana::lmap::decoded> media) {

}

