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
#include <string>

#include <boost/filesystem/path.hpp>

#include "aeon.hpp"
#include "file_util.hpp"

// This example needs aeon-service to be running

using nlohmann::json;
using std::cerr;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::stoi;
using std::string;

using nervana::loader;
using nervana::loader_factory;
using nervana::manifest_file;

void use_aeon(const string& address, int port, const boost::filesystem::path& manifest_path, const string& rdma_address, int rdma_port);

int main(int argc, char** argv)
{
    string address;
    int    port{0};
    boost::filesystem::path manifest_path;
    string rdma_address;
    int rdma_port{0};

    int opt;
    while ((opt = getopt(argc, argv, "a:p:m:r:s:h")) != EOF)
        switch (opt)
        {
        case 'a': address = optarg; break;
        case 'p': port    = stoi(optarg); break;
        case 'm': manifest_path = optarg; break;
        case 'r': rdma_address = optarg; break;
        case 's': rdma_port = stoi(optarg); break;
        case 'h':
        case '?': cout << "remote_iterator -a <address> -p <port> -m <manifest> -r <rdma_address> -s <rdma_port>" << endl; return 0;
        }

    if (address.empty() || port == 0 || manifest_path.empty())
    {
        cerr << "address (-a), port (-p) and manifest (-m) parameters have to be provided. Try remote_iterator -h for more information." << endl;
        return 1;
    }

    use_aeon(address, port, manifest_path, rdma_address, rdma_port);
}

void use_aeon(const string& address, int port, const boost::filesystem::path& manifest_path, const string& rdma_address, int rdma_port)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 4;

    json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label_config = {{"type", "label"}, {"binary", false}};
    json aug_config   = {{{"type", "image"}, {"flip_enable", true}}};
    json config       = {{"manifest_root", manifest_path.parent_path().string() },
                   {"manifest_filename", manifest_path.string() },
                   {"batch_size", batch_size},
                   {"iteration_mode", "ONCE"},
                   {"etl", {image_config, label_config}},
                   {"augmentation", aug_config},
                   {"remote", {{"address", address}, {"port", port}}}};

    if(!rdma_address.empty() && rdma_port)
    {
        config["remote"]["rdma_address"] = rdma_address;
        config["remote"]["rdma_port"] = rdma_port;
    }

    // initialize loader object
    loader_factory     factory;
    shared_ptr<loader> train_set = factory.get_loader(config);

    // retrieve dataset info
    cout << "batch size: " << train_set->batch_size() << endl;
    cout << "batch count: " << train_set->batch_count() << endl;
    cout << "record count: " << train_set->record_count() << endl;

    // iterate through all data
    int batch_no = 0;
    for (const auto& batch : *train_set)
    {
        cout << "\tbatch " << batch_no << " [number of elements: " << batch.size() << "]" << endl;
        batch_no++;
    }
}
