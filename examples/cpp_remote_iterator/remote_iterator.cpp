#include <iostream>
#include <string>

#include "aeon.hpp"
#include "file_util.hpp"

// This example needs aeon-server to be running

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

void use_aeon(const string& address, int port);

int main(int argc, char** argv)
{
    string address;
    int    port{0};

    int opt;
    while ((opt = getopt(argc, argv, "a:p:h")) != EOF)
        switch (opt)
        {
        case 'a': address = optarg; break;
        case 'p': port    = stoi(optarg); break;
        case 'h':
        case '?': cout << "remote_iterator -a <address> -p <port>" << endl; return 0;
        }

    if (address.empty() || port == 0)
    {
        cerr << "address (-a) and port (-p) parameters have to be provided" << endl;
        return 1;
    }

    use_aeon(address, port);
}

void use_aeon(const string& address, int port)
{
    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 4;
    string manifest_root = "../../../test/test_data/";
    string manifest_path = "../../../test/test_data/manifest.tsv";

    json image_config = {
        {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
    json label_config = {{"type", "label"}, {"binary", false}};
    json aug_config   = {{{"type", "image"}, {"flip_enable", true}}};
    json config       = {{"manifest_root", manifest_root},
                   {"manifest_filename", manifest_path},
                   {"batch_size", batch_size},
                   {"iteration_mode", "ONCE"},
                   {"etl", {image_config, label_config}},
                   {"augmentation", aug_config},
                   {"server", {{"address", address}, {"port", port}}}};

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
