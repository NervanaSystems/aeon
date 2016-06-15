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

#include <iostream>
#include <chrono>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/common.h>
    #include <libavutil/opt.h>
}

#include "gtest/gtest.h"

#include "imagegen.hpp"
#include "avgen.hpp"

using namespace std;

image_gen _datagen;

static void CreateImageDataset() {
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    _datagen.Directory("dataset")
            .Prefix("archive-")
            .MacrobatchMaxItems(500)
            // SetSize must be a multiple of (minibatchCount*batchSize) which is 8320 currently
            .DatasetSize(1500)
            .ImageSize(128,128)
            .Create();
    auto end = timer.now();
    cout << "datagen " << (chrono::duration_cast<chrono::milliseconds>(end - start)).count() << " msec" << endl;
}

static void CreateAudioDataset() {
    avgen::audio_encode("test1.mp2",2000);
}

static void CreateVideoDataset() {

}

static void DeleteDataset() {
    _datagen.Delete();
}

extern "C" int main( int argc, char** argv ) {
    CreateImageDataset();
    CreateAudioDataset();
    CreateVideoDataset();

    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();

    DeleteDataset();

    return rc;
}
