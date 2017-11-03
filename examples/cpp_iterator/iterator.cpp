/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <iostream>
#include <fstream>

#include "aeon.hpp"

using nlohmann::json;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::string;

using nervana::loader;
using nervana::loader_factory;
using nervana::manifest_file;

string generate_manifest_file(size_t record_count)
{
    string manifest_name = "manifest.txt";
    const char* image_files[] = {"flowers.jpg", "img_2112_70.jpg"};
    std::ofstream f(manifest_name);
    if (f)
    {
        f << manifest_file::get_metadata_char();
        f << manifest_file::get_file_type_id();
        f << manifest_file::get_delimiter();
        f << manifest_file::get_string_type_id();
        f << "\n";
        for (size_t i=0; i<record_count; i++)
        {
            f << image_files[i % 2];
            f << manifest_file::get_delimiter();
            f << std::to_string(i % 2);
            f << "\n";
        }
    }
    return manifest_name;
}

int main(int argc, char** argv)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 4;
    string manifest_root = "./";
    string manifest      = generate_manifest_file(20);

    json image_config = {{"type", "image"},
                               {"height", height},
                               {"width", width},
                               {"channel_major", false}};
    json label_config = {{"type", "label"},
                               {"binary", false}};
    json aug_config = {{{"type", "image"},
                             {"flip_enable", true}}};
    json config = {{"manifest_root", manifest_root},
                         {"manifest_filename", manifest},
                         {"batch_size", batch_size},
                         {"iteration_mode", "ONCE"},
                         {"etl", {image_config, label_config}},
                         {"augmentation", aug_config}};

    loader_factory factory;
    shared_ptr<loader> train_set = factory.get_loader( config );

    cout << "batch size: " << train_set->batch_size() << endl;
    cout << "batch count: " << train_set->batch_count() << endl;
    cout << "record count: " << train_set->record_count() << endl;

    int batch_no = 0;
    for(const auto& batch : *train_set)
    {
        cout << "\tbatch " << batch_no << " [number of elements: " << batch.size() << "]" << endl;
        batch_no++;
    }
}
