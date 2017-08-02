/*
 Copyright 2017 Nervana Systems Inc.
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "base64.hpp"
#include "log.hpp"
#include "service.hpp"

using namespace std;
using namespace nervana;

using ::testing::Return;
using nlohmann::json;

using nervana::fixed_buffer_map;

namespace
{
    const string session_id = "3";

    fixed_buffer_map& get_fixed_buffer_map();
    names_and_shapes  get_names_and_shapes();
}

class mock_http_connector : public http_connector
{
public:
    MOCK_METHOD2(get, http_response(const std::string& endpoint, const http_query_t& query));
    MOCK_METHOD2(post, http_response(const std::string& endpoint, const std::string& body));
    MOCK_METHOD2(post, http_response(const std::string& endpoint, const http_query_t& query));
};

TEST(service_connector, create_session)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        string config = "{}";
        json   expected_json;
        expected_json["status"]["type"] = "SUCCESS";
        expected_json["data"]["id"]     = session_id;
        auto expected_response          = http_response(http::status_created, expected_json.dump());
        EXPECT_CALL(*mock, post("/api/v1/dataset", config)).WillOnce(Return(expected_response));

        service_response<string> response = connector.create_session(config);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_EQ(response.data, session_id);
    }

    // status type is not SUCCESS
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        string config = "{}";
        json   expected_json;
        expected_json["status"]["type"]        = "FAILURE";
        expected_json["status"]["description"] = "something gone wrong";
        auto expected_response = http_response(http::status_created, expected_json.dump());
        EXPECT_CALL(*mock, post("/api/v1/dataset", config)).WillOnce(Return(expected_response));

        service_response<string> response = connector.create_session(config);

        EXPECT_EQ(response.status.type, service_status_type::FAILURE);
        EXPECT_EQ(response.status.description, "something gone wrong");
    }

    // ID is empty
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        string config = "{}";
        json   expected_json;
        expected_json["status"]["type"]        = "SUCCESS";
        expected_json["status"]["description"] = "all is ok but id is empty";
        auto expected_response = http_response(http::status_created, expected_json.dump());
        EXPECT_CALL(*mock, post("/api/v1/dataset", config)).WillOnce(Return(expected_response));

        ASSERT_THROW(connector.create_session(config), std::runtime_error);
    }
}

//TODO: write UTs for record_count, batch_count
TEST(service_connector, batch_size)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"]     = "SUCCESS";
        expected_json["data"]["batch_size"] = "64";
        auto   expected_response            = http_response(http::status_ok, expected_json.dump());
        string endpoint                     = "/api/v1/dataset/" + session_id + "/batch_size";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<int> response = connector.batch_size(session_id);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_EQ(response.data, 64);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "FAILURE";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/batch_size";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<int> response = connector.batch_size(session_id);

        EXPECT_EQ(response.status.type, service_status_type::FAILURE);
    }

    // batch_size has improper value
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"]        = "SUCCESS";
        expected_json["status"]["description"] = "some description";
        expected_json["data"]["batch_size"]    = "";
        auto   expected_response = http_response(http::status_ok, expected_json.dump());
        string endpoint          = "/api/v1/dataset/" + session_id + "/batch_size";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        ASSERT_THROW(connector.batch_size(session_id), std::runtime_error);
    }
}

TEST(service_connector, reset)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "SUCCESS";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/reset";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_status response = connector.reset(session_id);

        EXPECT_EQ(response.type, service_status_type::SUCCESS);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "FAILURE";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/reset";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_status response = connector.reset(session_id);

        EXPECT_EQ(response.type, service_status_type::FAILURE);
    }
}

TEST(service_connector, next)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        fixed_buffer_map& buffer_map = get_fixed_buffer_map();
        stringstream      serialized_buffer_map;
        buffer_map.serialize(serialized_buffer_map);
        std::vector<char> encoded_buffer_map = nervana::base64::encode(
            serialized_buffer_map.str().data(), serialized_buffer_map.str().size());

        json expected_json;
        expected_json["status"]["type"]   = "SUCCESS";
        expected_json["data"]["position"] = "2";
        expected_json["data"]["fixed_buffer_map"] =
            string(encoded_buffer_map.begin(), encoded_buffer_map.end());
        auto   expected_next_response = next_response(2, &buffer_map);
        auto   expected_response      = http_response(http::status_ok, expected_json.dump());
        string endpoint               = "/api/v1/dataset/" + session_id + "/next";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<next_response> response = connector.next(session_id);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_TRUE(response.data == expected_next_response);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "END_OF_DATASET";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/next";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<next_response> response = connector.next(session_id);

        EXPECT_EQ(response.status.type, service_status_type::END_OF_DATASET);
    }

    // batch_size has improper value
    /*{*/
    //auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
    //service_connector connector(mock);

    //json expected_json;
    //expected_json["status"]["type"]        = "SUCCESS";
    //expected_json["status"]["description"] = "some description";
    //expected_json["data"]["batch_size"]    = "";
    //auto   expected_response = http_response(http::status_ok, expected_json.dump());
    //string endpoint          = "/api/v1/dataset/" + session_id + "/batch_size";
    //EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

    //ASSERT_THROW(connector.batch_size(session_id), std::runtime_error);
    /*}*/
}

TEST(service_connector, names_and_shapes)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        map<string, shape_type> nas = get_names_and_shapes();
        stringstream serialized_nas;
        serialized_nas << nas;
        std::vector<char> encoded_nas =
            nervana::base64::encode(serialized_nas.str().data(), serialized_nas.str().size());

        json expected_json;
        expected_json["status"]["type"]           = "SUCCESS";
        expected_json["data"]["names_and_shapes"] = string(encoded_nas.begin(), encoded_nas.end());
        auto   expected_response = http_response(http::status_ok, expected_json.dump());
        string endpoint          = "/api/v1/dataset/" + session_id + "/names_and_shapes";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<names_and_shapes> response = connector.get_names_and_shapes(session_id);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_EQ(response.data, nas);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "FAILURE";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/names_and_shapes";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<names_and_shapes> response = connector.get_names_and_shapes(session_id);

        EXPECT_EQ(response.status.type, service_status_type::FAILURE);
    }
}

namespace
{
    fixed_buffer_map& get_fixed_buffer_map()
    {
        auto image_shape = shape_type(vector<size_t>{10, 10, 3}, output_type("uint8_t"));
        auto label_shape = shape_type(vector<size_t>{1}, output_type("uint32_t"));
        auto write_sizes =
            std::map<string, shape_type>{{"image", image_shape}, {"label", label_shape}};
        size_t                  batch_size = 1;
        static fixed_buffer_map result(write_sizes, batch_size);
        return result;
    }

    names_and_shapes get_names_and_shapes()
    {
        names_and_shapes nas;
        shape_type       s1{{1, 2}, {"int8_t"}};
        shape_type       s2{{1, 2, 3, 4, 5}, {"int32_t"}};
        nas["s1"] = s1;
        nas["s2"] = s2;
        return nas;
    }
}
