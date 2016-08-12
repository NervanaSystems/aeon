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

#include <fstream>
#include <sstream>

#include "json.hpp"
#include "manifest_nds.hpp"

using namespace std;

NDSManifest::NDSManifest(const std::string filename) {
    auto j = nlohmann::json::parse(filename);

    baseurl = j["baseurl"];
    token = j["params"]["token"];
    collection_id = j["params"]["collection_id"];
}

NDSManifest::~NDSManifest() {
}

string NDSManifest::hash() {
    stringstream contents;
    contents << baseurl << collection_id;
    std::size_t h = std::hash<std::string>()(contents.str());
    stringstream ss;
    ss << std::hex << h;
    return ss.str();
}

string NDSManifest::version() {
    // NDS Manifest doesn't have versions since collections are immutable
    return "";
}

bool NDSManifest::isLikelyJSON(const std::string filename) {
    // check the first character of the file to see if it is a json
    // object.  If so, we want to parse this as an NDS Manifest
    // instead of a CSV
    ifstream f(filename);
    char first_char;

    f.get(first_char);

    return first_char == '{';
}
