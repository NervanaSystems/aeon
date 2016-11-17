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

#pragma once

#include <string>

#include "manifest.hpp"

namespace nervana
{
    class manifest_nds;
}

class nervana::manifest_nds : public nervana::manifest
{
public:
    manifest_nds(const std::string& filename);
    ~manifest_nds() {}
    std::string cache_id() override;

    // NDS manifests doesn't have versions since collections are immutable
    std::string version() override { return ""; }
    static bool is_likely_json(const std::string filename);

    std::string baseurl;
    std::string token;
    int         collection_id;
};
