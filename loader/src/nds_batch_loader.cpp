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

// inspired by https://techoverflow.net/blog/2013/03/15/c-simple-http-download-using-libcurl-easy-api/

// apt-get install libcurl4-openssl-dev
#include <curl/curl.h>
#include <curl/easy.h>
#include <curl/curlbuild.h>

#include "nds_batch_loader.hpp"

using namespace std;

size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream) {
    // callback used by curl.  writes data from ptr into the
    // stringstream passed in to `stream`.

    string data((const char*) ptr, (size_t) size * nmemb);
    *((stringstream*) stream) << data << endl;
    return size * nmemb;
}

NDSBatchLoader::NDSBatchLoader(const std::string baseurl, int tag_id)
    : _baseurl(baseurl), _tag_id(tag_id) {
    // reuse curl connection across requests
    _curl = curl_easy_init();
}

NDSBatchLoader::~NDSBatchLoader() {
    curl_easy_cleanup(_curl);
}

void NDSBatchLoader::loadBlock(BufferPair &dest, uint block_num, uint block_size) {
    // not much use in mutlithreading here since in most cases, our next step is
    // to shuffle the entire BufferPair, which requires the entire buffer loaded.

    // get data from url and write it into cpio_stream
    stringstream cpio_stream;

    curl_easy_setopt(_curl, CURLOPT_URL, url(block_num, block_size).c_str());
    curl_easy_setopt(_curl, CURLOPT_FOLLOWLOCATION, 1L);
    // Prevent "longjmp causes uninitialized stack frame" bug
    curl_easy_setopt(_curl, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(_curl, CURLOPT_ACCEPT_ENCODING, "deflate");
    curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(_curl, CURLOPT_WRITEDATA, &_cpio_stream);

    // Perform the request, res will get the return code
    CURLcode res = curl_easy_perform(_curl);

    // Check for errors
    if (res != CURLE_OK) {
        stringstream ss;
        ss << "curl_easy_perform() failed: ";
        ss << curl_easy_strerror(res);
        throw std::runtime_error(ss.str());
    }

    // parse cpio stream into dest one item at a time
    CPIOReader reader(&_cpio_stream);
    for(int i=0; i < reader.itemCount(); ++i) {
        reader.read(*dest.first);
        reader.read(*dest.second);
    }
}

const string NDSBatchLoader::url(uint block_num, uint block_size) {
    stringstream ss;
    ss << _baseurl << "/macrobatch?";
    ss << "macro_batch_index=" << block_num;
    ss << "&macro_batch_max_size=" << block_size;
    ss << "&tag_id=" << _tag_id;
    return ss.str();
}

uint NDSBatchLoader::objectCount() {
    // TODO
}

uint NDSBatchLoader::blockCount(uint block_size) {
    // TODO
}
