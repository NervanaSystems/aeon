/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <fstream>
#include <numeric>
#include <sstream>

#include <curl/curl.h>
#include <curl/easy.h>

#include "json.hpp"
#include "manifest_nds.hpp"
#include "interface.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

network_client::network_client(const std::string& baseurl,
                               const std::string& token,
                               size_t             collection_id,
                               size_t             block_size,
                               size_t             shard_count,
                               size_t             shard_index)
    : m_baseurl(baseurl)
    , m_token(token)
    , m_collection_id(collection_id)
    , m_shard_count(shard_count)
    , m_shard_index(shard_index)
    , m_macrobatch_size(block_size)
{
    curl_global_init(CURL_GLOBAL_ALL);
}

network_client::~network_client()
{
    curl_global_cleanup();
}

size_t network_client::callback(void* ptr, size_t size, size_t nmemb, void* stream)
{
    stringstream& ss = *(stringstream*)stream;
    // callback used by curl.  writes data from ptr into the
    // stringstream passed in to `stream`.
    ss.write((const char*)ptr, size * nmemb);
    return size * nmemb;
}

void network_client::get(const string& url, stringstream& stream)
{
    // reuse curl connection across requests
    void* m_curl = curl_easy_init();

    // given a url, make an HTTP GET request and fill stream with
    // the body of the response

    curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, callback);
    curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(m_curl, CURLOPT_NOPROXY, "127.0.0.1,localhost");

    // Perform the request, res will get the return code
    CURLcode res = curl_easy_perform(m_curl);

    // Check for errors
    long http_code = 0;
    curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200 || res != CURLE_OK)
    {
        stringstream ss;
        ss << "HTTP GET on \n'" << url << "' failed. ";
        ss << "status code: " << http_code;
        if (res != CURLE_OK)
        {
            ss << " curl return: " << curl_easy_strerror(res);
        }

        curl_easy_cleanup(m_curl);
        throw std::runtime_error(ss.str());
    }

    curl_easy_cleanup(m_curl);
}

string network_client::load_block_url(size_t block_num)
{
    stringstream ss;
    ss << m_baseurl << "/macrobatch/?";
    ss << "macro_batch_index=" << block_num;
    ss << "&macro_batch_max_size=" << m_macrobatch_size;
    ss << "&collection_id=" << m_collection_id;
    ss << "&shard_count=" << m_shard_count;
    ss << "&shard_index=" << m_shard_index;
    ss << "&token=" << m_token;
    return ss.str();
}

string network_client::metadata_url()
{
    stringstream ss;
    ss << m_baseurl << "/object_count/?";
    ss << "macro_batch_max_size=" << m_macrobatch_size;
    ss << "&collection_id=" << m_collection_id;
    ss << "&shard_count=" << m_shard_count;
    ss << "&shard_index=" << m_shard_index;
    ss << "&token=" << m_token;
    return ss.str();
}

manifest_nds_builder& manifest_nds_builder::filename(const std::string& filename)
{
    parse_json(filename);
    return *this;
}

manifest_nds_builder& manifest_nds_builder::base_url(const std::string& url)
{
    m_base_url = url;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::token(const std::string& token)
{
    m_token = token;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::collection_id(size_t collection_id)
{
    m_collection_id = collection_id;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::block_size(size_t block_size)
{
    m_block_size = block_size;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::elements_per_record(size_t elements_per_record)
{
    m_elements_per_record = elements_per_record;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::shard_count(size_t shard_count)
{
    m_shard_count = shard_count;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::shard_index(size_t shard_index)
{
    m_shard_index = shard_index;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::shuffle(bool enable)
{
    m_shuffle = enable;
    return *this;
}

manifest_nds_builder& manifest_nds_builder::seed(uint32_t seed)
{
    m_seed = seed;
    return *this;
}

void manifest_nds_builder::parse_json(const std::string& filename)
{
    // parse json
    nlohmann::json j;
    try
    {
        ifstream ifs(filename);
        ifs >> j;
    }
    catch (const std::exception& ex)
    {
        stringstream ss;
        ss << "Error while parsing manifest json: " << filename << " : ";
        ss << ex.what();
        throw std::runtime_error(ss.str());
    }

    // extract manifest params from parsed json
    try
    {
        interface::config::parse_value(m_base_url, "url", j, interface::config::mode::REQUIRED);

        auto val = j.find("params");
        if (val != j.end())
        {
            nlohmann::json params = *val;
            interface::config::parse_value(
                m_token, "token", params, interface::config::mode::REQUIRED);
            interface::config::parse_value(
                m_collection_id, "collection_id", params, interface::config::mode::REQUIRED);
        }
        else
        {
            throw std::runtime_error("couldn't find key 'params' in nds manifest file.");
        }
    }
    catch (const std::exception& ex)
    {
        stringstream ss;
        ss << "Error while pulling config out of manifest json: " << filename << " : ";
        ss << ex.what();
        throw std::runtime_error(ss.str());
    }
}

manifest_nds manifest_nds_builder::create()
{
    if (m_base_url == "")
    {
        throw invalid_argument("base_url is required");
    }
    if (m_token == "")
    {
        throw invalid_argument("token is required");
    }
    if (m_collection_id == -1)
    {
        throw invalid_argument("collection_id is required");
    }
    if (m_elements_per_record == -1)
    {
        throw invalid_argument("elements_per_record is required");
    }

    return manifest_nds(m_base_url,
                        m_token,
                        m_collection_id,
                        m_block_size,
                        m_elements_per_record,
                        m_shard_count,
                        m_shard_index,
                        m_shuffle,
                        m_seed);
}

std::shared_ptr<manifest_nds> manifest_nds_builder::make_shared()
{
    return std::shared_ptr<manifest_nds>(new manifest_nds(m_base_url,
                                                          m_token,
                                                          m_collection_id,
                                                          m_block_size,
                                                          m_elements_per_record,
                                                          m_shard_count,
                                                          m_shard_index,
                                                          m_shuffle,
                                                          m_seed));
}

manifest_nds::manifest_nds(const std::string& base_url,
                           const std::string& token,
                           size_t             collection_id,
                           size_t             block_size,
                           size_t             elements_per_record,
                           size_t             shard_count,
                           size_t             shard_index,
                           bool               enable_shuffle,
                           uint32_t           seed)
    : m_base_url(base_url)
    , m_token(token)
    , m_collection_id(collection_id)
    , m_elements_per_record(elements_per_record)
    , m_network_client{base_url, token, collection_id, block_size, shard_count, shard_index}
    , m_current_block_number{0}
    , m_shuffle{enable_shuffle}
    , m_rnd{seed ? seed : random_device{}()}
{
    load_metadata();

    m_block_load_sequence.reserve(m_block_count);
    m_block_load_sequence.resize(m_block_count);
    iota(m_block_load_sequence.begin(), m_block_load_sequence.end(), 0);

    if (m_shuffle)
    {
        shuffle(m_block_load_sequence.begin(), m_block_load_sequence.end(), m_rnd);
    }

    // // parse json
    // nlohmann::json j;
    // try
    // {
    //     ifstream ifs(filename);
    //     ifs >> j;
    // }
    // catch (std::exception& ex)
    // {
    //     stringstream ss;
    //     ss << "Error while parsing manifest json: " << filename << " : ";
    //     ss << ex.what();
    //     throw std::runtime_error(ss.str());
    // }

    // // extract manifest params from parsed json
    // try
    // {
    //     interface::config::parse_value(m_baseurl, "url", j, interface::config::mode::REQUIRED);

    //     auto val = j.find("params");
    //     if (val != j.end())
    //     {
    //         nlohmann::json params = *val;
    //         interface::config::parse_value(m_token, "token", params, interface::config::mode::REQUIRED);
    //         interface::config::parse_value(m_collection_id, "collection_id", params, interface::config::mode::REQUIRED);
    //     }
    //     else
    //     {
    //         throw std::runtime_error("couldn't find key 'params' in nds manifest file.");
    //     }
    // }
    // catch (std::exception& ex)
    // {
    //     stringstream ss;
    //     ss << "Error while pulling config out of manifest json: " << filename << " : ";
    //     ss << ex.what();
    //     throw std::runtime_error(ss.str());
    // }
}

encoded_record_list* manifest_nds::next()
{
    encoded_record_list* rc = nullptr;

    if (m_current_block_number < m_block_count)
    {
        rc = load_block(m_block_load_sequence[m_current_block_number]);

        m_current_block_number++;
    }

    return rc;
}

encoded_record_list* manifest_nds::load_block(size_t block_index)
{
    // not much use in mutlithreading here since in most cases, our next step is
    // to shuffle the entire BufferPair, which requires the entire buffer loaded.
    m_current_block.clear();

    // get data from url and write it into cpio_stream
    stringstream stream;
    string       url = m_network_client.load_block_url(block_index);
    m_network_client.get(url, stream);

    // parse cpio_stream into dest one record (consisting of multiple elements) at a time
    nervana::cpio::reader reader(stream);
    size_t                record_count = reader.record_count();
    for (size_t record_number = 0; record_number < record_count; record_number++)
    {
        encoded_record record;
        for (size_t element = 0; element < m_elements_per_record; element++)
        {
            vector<char> buffer;
            string       filename = reader.read(buffer);
            if (filename == cpio::CPIO_TRAILER || filename == cpio::AEON_TRAILER)
            {
                break;
            }
            record.add_element(buffer);
        }
        m_current_block.add_record(record);
    }
    return &m_current_block;
}

string manifest_nds::cache_id()
{
    stringstream contents;
    contents << m_base_url << m_collection_id;
    std::size_t  h = std::hash<std::string>()(contents.str());
    stringstream ss;
    ss << std::hex << h;
    return ss.str();
}

bool manifest_nds::is_likely_json(const std::string filename)
{
    // check the first character of the file to see if it is a json
    // object.  If so, we want to parse this as an NDS Manifest
    // instead of a CSV
    ifstream f(filename);
    char     first_char;

    f.get(first_char);

    return first_char == '{';
}

size_t manifest_nds::write_data(void* ptr, size_t size, size_t nmemb, void* stream)
{
    stringstream& ss = *(stringstream*)stream;
    // callback used by curl.  writes data from ptr into the
    // stringstream passed in to `stream`.

    ss.write((const char*)ptr, size * nmemb);
    return size * nmemb;
}

void manifest_nds::load_metadata()
{
    // fetch metadata and store in local attributes

    stringstream json_stream;
    string       url = m_network_client.metadata_url();
    m_network_client.get(url, json_stream);
    string         json_str = json_stream.str();
    nlohmann::json metadata;

    try
    {
        metadata = nlohmann::json::parse(json_str);
    }
    catch (const std::exception& ex)
    {
        stringstream ss;
        ss << "exception parsing metadata from nds ";
        ss << url << " " << ex.what() << " ";
        ss << json_str;
        throw std::runtime_error(ss.str());
    }

    nervana::interface::config::parse_value(
        m_record_count, "record_count", metadata, nervana::interface::config::mode::REQUIRED);
    nervana::interface::config::parse_value(m_block_count,
                                            "macro_batch_per_shard",
                                            metadata,
                                            nervana::interface::config::mode::REQUIRED);
}
