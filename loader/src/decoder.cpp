#include <iostream>
#include <fstream>
#include <vector>
#include <json.hpp>


using json = nlohmann::json;

int main (int argc, char **argv) {
    json o;
    std::ifstream ifile(argv[1]);
    ifile >> o;
    json img = o["imagenet"];
    for (auto it = img.begin(); it != img.end(); ++it) {
      std::cout << it.key() << " : " << it.value() << "\n";
    }

    // // Just get a single item
    // auto dpair = bf.readItem();
    // bf.close();
    // ByteVect data = *(dpair.first);
    // ByteVect labels = *(dpair.second);

    // int label_idx = *reinterpret_cast<int *>(&labels[0]);
    // // We'll do 10 decodings of the same image;
    // Image decoder(imgp, iip, 0);
    // int num_decode = 10;
    // int num_pixels = imgp->getSize().area() * 3;
    // ByteVect outbuf(num_pixels * num_decode);
    // std::cout << "numpixels: " << num_pixels << std::endl;
    // std::cout << "outbuf size: " << outbuf.size() << std::endl;
    // std::cout << "label index: " << label_idx << std::endl;

    // for (int i = 0; i < num_decode; i++) {
    //     decoder.transform(&data[0], data.size(), &outbuf[i * num_pixels], num_pixels);
    // }

    // std::ofstream file (argv[2], std::ofstream::out | std::ofstream::binary);
    // file.write((char *) &num_decode, sizeof(int));
    // file.write((char *) &num_pixels, sizeof(int));
    // file.write((char *) &outbuf[0], outbuf.size());
    // file.close();
    // delete imgp;
    // delete iip;
}


