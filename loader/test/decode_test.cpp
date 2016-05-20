#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include "buffer.hpp"
#include "batchfile.hpp"
#include "image.hpp"
#include "datagen.hpp"
#include "gtest/gtest.h"

extern DataGen _datagen;

using namespace std;

TEST(loader,decode) {
    ImageParams *imgp = new ImageParams(3, 128, 128, false, true, // channels, h, w, augment, flip
        20, 100,   // Scale Params
        60, 140,    // Contrast params
        -10, 10,       // Rotation params
        133,          // Aspect Ratio
        false, 0, 0, 0, 0);  // subtract mean, r, g, b, gray means
    ImageIngestParams *iip = new ImageIngestParams(true, true, 256, 256);

    BatchFile bf;
    string batchFileName = _datagen.GetDatasetPath() + "/archive-0.cpio";
    bf.openForRead(batchFileName);

    // Just get a single item
    auto dpair = bf.readItem();
    bf.close();
    ByteVect data = *(dpair.first);
    ByteVect labels = *(dpair.second);

    int label_idx = *reinterpret_cast<int *>(&labels[0]);
    // We'll do 10 decodings of the same image;
    Image decoder(imgp, iip, 0);
    int num_decode = 10;
    int num_pixels = imgp->getSize().area() * 3;
    ByteVect outbuf(num_pixels * num_decode);
    std::cout << "numpixels: " << num_pixels << std::endl;
    std::cout << "outbuf size: " << outbuf.size() << std::endl;
    std::cout << "label index: " << label_idx << std::endl;
    EXPECT_EQ( 42, label_idx ) << "Label mismatch";

    for (int i = 0; i < num_decode; i++) {
        decoder.transform(&data[0], data.size(), &outbuf[i * num_pixels], num_pixels);
    }

    // std::ofstream file (argv[2], std::ofstream::out | std::ofstream::binary);
    // file.write((char *) &num_decode, sizeof(int));
    // file.write((char *) &num_pixels, sizeof(int));
    // file.write((char *) &outbuf[0], outbuf.size());
    // file.close();
    delete imgp;
    delete iip;
}
