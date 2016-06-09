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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "manifest.hpp"

using namespace std;

Manifest::Manifest(string filename)
: _filename(filename) {
    // for now parse the entire manifest on creation
    parse();
}

string Manifest::hash() {
    // TODO
    return "42";
}

void Manifest::parse() {
    // TODO: handle case where file doesn't exist
    ifstream infile(_filename);

    parseStream(infile);
}

size_t Manifest::getSize() const {
    return _filename_pairs.size();
}

void Manifest::parseStream(istream& is) {
    // parse istream is and load the entire thing into _filename_pairs
    string line;
    pair<string, string> filename_pair;

    // TODO: could probably do this more cleanly ...
    while(std::getline(is, line)) {
        istringstream lineis(line);
        std::getline(lineis, filename_pair.first, ',');
        std::getline(lineis, filename_pair.second);

        _filename_pairs.push_back(filename_pair);
    }

    // TODO: only optionally shuffle.  If we don't need to shuffle,
    // there may be small performance benefits in some situations to
    // stream the filename_pairs instead of loading them all at once.
    // That said, in the event that there is no cache and we are
    // resuming training at a specific epoch, we may need to be able to
    // jump around and read random blocks of the file, so a purely
    // stream based interface is not sufficient.
    shuffleFilenamePairs();
}

void Manifest::shuffleFilenamePairs() {
    // shuffles _filename_pairs.  It is possible that the order of the
    // filenames in the manifest file were in some sorted order and we
    // don't want our blocks to be biased by that order.
    std::random_device rd;
    std::shuffle(_filename_pairs.begin(), _filename_pairs.end(), std::mt19937(rd()));
}

Manifest::iter Manifest::begin() const {
    return _filename_pairs.begin();
}

Manifest::iter Manifest::end() const {
    return _filename_pairs.end();
}
