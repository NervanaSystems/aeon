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

#include "service.hpp"

using namespace std;
using namespace nervana;

using nlohmann::json;
using ::testing::Return;

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
        expected_json["status"]["type"]        = "SUCCESS";
        expected_json["data"]["id"]            = "123";
        auto expected_response = http_response(http::status_created, expected_json.dump());
        EXPECT_CALL(*mock, post("/api/v1/dataset", config)).WillOnce(Return(expected_response));

        service_response<string> response = connector.create_session(config);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_EQ(response.data, "123");
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
