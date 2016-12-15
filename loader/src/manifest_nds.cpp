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

#include <fstream>
#include <sstream>

#include <curl/curl.h>
#include <curl/easy.h>
#include <curl/curlbuild.h>

#include "json.hpp"
#include "manifest_nds.hpp"
#include "interface.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

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

    return manifest_nds(m_base_url, m_token, m_collection_id, m_block_size, m_elements_per_record, m_shard_count, m_shard_index);
}


manifest_nds::manifest_nds(const std::string& base_url, const std::string& token, size_t collection_id, size_t block_size,
                           size_t elements_per_record, size_t shard_count, size_t shard_index)
    : m_base_url(base_url)
    , m_token(token)
    , m_collection_id(collection_id)
    , m_block_size(block_size)
    , m_elements_per_record(elements_per_record)
    , m_shard_count(shard_count)
    , m_shard_index(shard_index)
{
    load_metadata();
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
    // if (m_counter < m_block_list.size())
    // {
    //     rc = &(m_block_list[m_counter]);
    //     m_counter++;
    // }
    return rc;
}

encoded_record_list manifest_nds::load_block(size_t block_index)
{
    // not much use in mutlithreading here since in most cases, our next step is
    // to shuffle the entire BufferPair, which requires the entire buffer loaded.
    encoded_record_list rc;

    // get data from url and write it into cpio_stream
    stringstream stream;
    get(load_block_url(block_index), stream);

    // parse cpio_stream into dest one record (consisting of multiple elements) at a time
    nervana::cpio::reader reader(stream);
    size_t record_count = reader.record_count();
    for (size_t record_number=0; record_number<record_count; record_number++)
    {
        encoded_record record;
        for (size_t element=0; element<m_elements_per_record; element++)
        {
            vector<char> buffer;
            string filename = reader.read(buffer);
            if (filename == cpio::CPIO_TRAILER || filename == cpio::AEON_TRAILER)
            {
                break;
            }
            record.add_element(buffer);
        }
        rc.add_record(record);
    }
    return rc;
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

void manifest_nds::get(const string& url, stringstream& stream)
{
    // reuse curl connection across requests
    void* m_curl = curl_easy_init();

    // given a url, make an HTTP GET request and fill stream with
    // the body of the response

    curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &stream);
    // curl_easy_setopt(m_curl, CURLOPT_VERBOSE, 1L);

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

const string manifest_nds::load_block_url(size_t block_index)
{
    stringstream ss;
    ss << m_base_url << "/macrobatch/?";
    ss << "macro_batch_index=" << block_index;
    ss << "&macro_batch_max_size=" << m_block_size;
    ss << "&collection_id=" << m_collection_id;
    ss << "&shard_count=" << m_shard_count;
    ss << "&shard_index=" << m_shard_index;
    ss << "&token=" << m_token;
    return ss.str();
}

const string manifest_nds::metadata_url()
{
    stringstream ss;
    ss << m_base_url << "/object_count/?";
    ss << "macro_batch_max_size=" << m_block_size;
    ss << "&collection_id=" << m_collection_id;
    ss << "&shard_count=" << m_shard_count;
    ss << "&shard_index=" << m_shard_index;
    ss << "&token=" << m_token;
    return ss.str();
}

void manifest_nds::load_metadata()
{
    // fetch metadata and store in local attributes

    stringstream json_stream;
    get(metadata_url(), json_stream);
    string json_str = json_stream.str();
    nlohmann::json metadata;

    try
    {
        metadata = nlohmann::json::parse(json_str);
    }
    catch (std::exception& ex)
    {
        stringstream ss;
        ss << "exception parsing metadata from nds ";
        ss << metadata_url() << " " << ex.what() << " ";
        ss << json_str;
        throw std::runtime_error(ss.str());
    }

    nervana::interface::config::parse_value(m_record_count, "record_count", metadata, nervana::interface::config::mode::REQUIRED);
    nervana::interface::config::parse_value(m_block_count, "macro_batch_per_shard", metadata,
                                            nervana::interface::config::mode::REQUIRED);
}
