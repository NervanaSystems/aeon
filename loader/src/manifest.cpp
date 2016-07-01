/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "manifest.hpp"

using namespace std;

const int makeRandomSeed() {
    // helper for deligated constructor below
    std::random_device rd;
    return rd();
}

Manifest::Manifest(string filename, bool shuffle, const int randomSeed)
: _filename(filename), _shuffle(shuffle), _randomSeed(randomSeed) {
    // for now parse the entire manifest on creation
    parse();
}

Manifest::Manifest(string filename, bool shuffle)
    : Manifest(filename, shuffle, makeRandomSeed()) {
}

string Manifest::hash() {
    // returns a hash of the _filename
    std::size_t h = std::hash<std::string>()(_filename);
    stringstream ss;
    ss << std::hex << h;
    return ss.str();
}

string Manifest::version() {
    // return the manifest version.  In this case it is just the timestamp
    // on the file
    struct stat stats;
    int result = stat(_filename.c_str(), &stats);
    if (result == -1) {
        stringstream ss;
        ss << "Could not find manifest file " << _filename;
        throw std::runtime_error(ss.str());
    }

    stringstream ss;
    ss << stats.st_mtime;
    return ss.str();
}

void Manifest::parse() {
    ifstream infile(_filename);

    if(!infile.is_open()) {
        stringstream ss;
        ss << "Manifest file " << _filename << " doesnt exit.";
        throw std::runtime_error(ss.str());
    }

    parseStream(infile);
}

size_t Manifest::getSize() const {
    return _filename_pairs.size();
}

void Manifest::parseStream(istream& is) {
    // parse istream is and load the entire thing into _filename_pairs
    string line;
    pair<string, string> filename_pair;

    // read in each line, then from that istringstream, read in up to
    // the comma, then everything after.
    while(std::getline(is, line)) {
        istringstream lineis(line);
        std::getline(lineis, filename_pair.first, ',');
        std::getline(lineis, filename_pair.second);

        _filename_pairs.push_back(filename_pair);
    }

    // If we don't need to shuffle, there may be small performance
    // benefits in some situations to stream the filename_pairs instead
    // of loading them all at once.  That said, in the event that there
    // is no cache and we are resuming training at a specific epoch, we
    // may need to be able to jump around and read random blocks of the
    // file, so a purely stream based interface is not sufficient.
    if(_shuffle) {
        shuffleFilenamePairs();
    }
}

void Manifest::shuffleFilenamePairs() {
    // shuffles _filename_pairs.  It is possible that the order of the
    // filenames in the manifest file were in some sorted order and we
    // don't want our blocks to be biased by that order.
    std::shuffle(_filename_pairs.begin(), _filename_pairs.end(), std::mt19937(_randomSeed));
}

Manifest::iter Manifest::begin() const {
    return _filename_pairs.begin();
}

Manifest::iter Manifest::end() const {
    return _filename_pairs.end();
}
