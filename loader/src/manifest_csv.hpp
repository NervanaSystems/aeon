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

#include <vector>
#include <string>
#include <random>

#include "manifest.hpp"

/* Manifest
 *
 * load a manifest file and parse the filenames
 *
 * The format of the file should be something like:
 *
 * object_filename1,target_filename1
 * object_filename2,target_filename2
 * ...
 *
 * string hash() is to be used as a key for the cache.  It is possible
 * that it will be better to use the filename and last modified time as
 * a key instead.
 *
 */
namespace nervana {

    class manifest_csv : public manifest {
    public:
        manifest_csv(const std::string& filename, bool shuffle, const std::string& root = "");

        typedef std::vector<std::string> FilenameList;
        typedef std::vector<FilenameList>::const_iterator iter;

        std::string cache_id();
        std::string version();
        size_t objectCount() const { return _filename_lists.size(); }

        // begin and end provide iterators over the FilenameLists
        iter begin() const { return _filename_lists.begin(); }
        iter end() const { return _filename_lists.end(); }

        void generate_subset(float subset_fraction);

    protected:
        void parse_stream(std::istream& is);
        void shuffle_filename_lists();

    private:
        const std::string _filename;
        const std::string _root;
        const bool _shuffle;
        std::vector<FilenameList> _filename_lists;
    };
}
