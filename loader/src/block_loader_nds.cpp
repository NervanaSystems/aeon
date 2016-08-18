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

#include "json.hpp"
#include "block_loader_nds.hpp"
#include "interface.hpp"

using namespace std;
using namespace nervana;

size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream) {
    // callback used by curl.  writes data from ptr into the
    // stringstream passed in to `stream`.

    string data((const char*) ptr, (size_t) size * nmemb);
    *((stringstream*) stream) << data;
    return size * nmemb;
}

block_loader_nds::block_loader_nds(const std::string baseurl, const std::string token, int collection_id, uint block_size, int shard_count, int shard_index)
    : block_loader(block_size), _baseurl(baseurl), _token(token), _collection_id(collection_id),
      _shard_count(shard_count), _shard_index(shard_index) {
    assert(shard_index < shard_count);

    loadMetadata();
}

block_loader_nds::~block_loader_nds() {
}

void block_loader_nds::loadBlock(nervana::buffer_in_array& dest, uint block_num) {
    // not much use in mutlithreading here since in most cases, our next step is
    // to shuffle the entire BufferPair, which requires the entire buffer loaded.

    // get data from url and write it into cpio_stream
    stringstream cpio_stream;
    get(loadBlockURL(block_num), cpio_stream);

    // parse cpio_stream into dest one record (consisting of multiple elements) at a time
    nervana::cpio::reader reader(&cpio_stream);
    for(int i=0; i < reader.itemCount() / dest.size(); ++i) {
        for (auto d: dest) {
            reader.read(*d);
        }
    }
}

void block_loader_nds::get(const string url, stringstream &stream) {
    // reuse curl connection across requests
    _curl = curl_easy_init();

    // given a url, make an HTTP GET request and fill stream with
    // the body of the response

    curl_easy_setopt(_curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(_curl, CURLOPT_FOLLOWLOCATION, 1L);
    // Prevent "longjmp causes uninitialized stack frame" bug
    curl_easy_setopt(_curl, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(_curl, CURLOPT_ACCEPT_ENCODING, "deflate");
    curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(_curl, CURLOPT_WRITEDATA, &stream);

    // Perform the request, res will get the return code
    CURLcode res = curl_easy_perform(_curl);

    // Check for errors
    if (res != CURLE_OK) {
        long http_code = 0;
        curl_easy_getinfo (_curl, CURLINFO_RESPONSE_CODE, &http_code);

        stringstream ss;
        ss << "HTTP GET on " << url << "failed. ";
        ss << "status code: " << http_code << ". ";
        ss << curl_easy_strerror(res);

        curl_easy_cleanup(_curl);
        throw std::runtime_error(ss.str());
    }

    curl_easy_cleanup(_curl);
}

const string block_loader_nds::loadBlockURL(uint block_num) {
    stringstream ss;
    ss << _baseurl << "/macrobatch/?";
    ss << "macro_batch_index=" << block_num;
    ss << "&macro_batch_max_size=" << _block_size;
    ss << "&collection_id=" << _collection_id;
    ss << "&shard_count=" << _shard_count;
    ss << "&shard_index=" << _shard_index;
    ss << "&token=" << _token;
    return ss.str();
}

const string block_loader_nds::metadataURL() {
    stringstream ss;
    ss << _baseurl << "/object_count/?";
    ss << "macro_batch_max_size=" << _block_size;
    ss << "&collection_id=" << _collection_id;
    ss << "&shard_count=" << _shard_count;
    ss << "&shard_index=" << _shard_index;
    ss << "&token=" << _token;
    return ss.str();
}

void block_loader_nds::loadMetadata() {
    // fetch metadata and store in local attributes

    stringstream json_stream;
    get(metadataURL(), json_stream);
    string json_str = json_stream.str();
    nlohmann::json metadata;
    try {
        metadata = nlohmann::json::parse(json_str);
    } catch (std::exception& ex) {
        stringstream ss;
        ss << "exception parsing metadata from nds ";
        ss << metadataURL() << " " << ex.what() << " ";
        ss << json_str;
        cout << "json_str: " << json_str << endl;
        throw std::runtime_error(ss.str());
    }

    nervana::interface::config::parse_value(_objectCount, "record_count", metadata, nervana::interface::config::mode::REQUIRED);
    nervana::interface::config::parse_value(_blockCount, "macro_batch_per_shard", metadata, nervana::interface::config::mode::REQUIRED);
}

uint block_loader_nds::objectCount() {
    return _objectCount;
}

uint block_loader_nds::blockCount() {
    return _blockCount;
}
