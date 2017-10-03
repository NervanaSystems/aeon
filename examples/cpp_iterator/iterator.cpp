#include <iostream>
#include <fstream>

#include "aeon.hpp"

std::string generate_manifest_file(size_t record_count)
{
    std::string manifest_name = "manifest.txt";
    const char* image_files[] = {"flowers.jpg", "img_2112_70.jpg"};
    std::ofstream f(manifest_name);
    if (f)
    {
        f << nervana::manifest_file::get_metadata_char();
        f << nervana::manifest_file::get_file_type_id();
        f << nervana::manifest_file::get_delimiter();
        f << nervana::manifest_file::get_string_type_id();
        f << "\n";
        for (size_t i=0; i<record_count; i++)
        {
            f << image_files[i % 2];
            f << nervana::manifest_file::get_delimiter();
            f << std::to_string(i % 2);
            f << "\n";
        }
    }
    return manifest_name;
}

int main(int argc, char** argv)
{
    std::cout << "hello world" << std::endl;

    int    height        = 32;
    int    width         = 32;
    size_t batch_size    = 1;
    std::string manifest_root = "";
    std::string manifest      = generate_manifest_file(20);

    nlohmann::json image_config = {{"type", "image"},
                               {"height", height},
                               {"width", width},
                               {"channel_major", false}};
    nlohmann::json label_config = {{"type", "label"},
                               {"binary", false}};
    nlohmann::json aug_config = {{"type", "image"},
                             {"flip_enable", true}};
    nlohmann::json config = {{"manifest_root", manifest_root},
                         {"manifest_filename", manifest},
                         {"batch_size", batch_size},
                         {"iteration_mode", "INFINITE"},
                         {"etl", {image_config, label_config}},
                         {"augmentation", {{aug_config}}}};

    auto train_set = nervana::loader{config};
}
