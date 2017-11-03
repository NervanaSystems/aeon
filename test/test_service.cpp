/*
 Copyright 2017 Intel(R) Nervana(TM)
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
#include "helpers.hpp"
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

    class mock_http_connector : public http_connector
    {
    public:
        MOCK_METHOD2(get, http_response(const std::string& endpoint, const http_query_t& query));
        MOCK_METHOD2(post, http_response(const std::string& endpoint, const std::string& body));
        MOCK_METHOD2(post, http_response(const std::string& endpoint, const http_query_t& query));
        MOCK_METHOD2(del, http_response(const std::string& endpoint, const http_query_t& query));
    };

    enum class property
    {
        batch_count,
        batch_size,
        record_count
    };

    template <property P>
    void test_get_property(const string& property);
}

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

TEST(service_connector, batch_size)
{
    test_get_property<property::batch_size>("batch_size");
}

TEST(service_connector, record_count)
{
    test_get_property<property::record_count>("record_count");
}

TEST(service_connector, batch_count)
{
    test_get_property<property::batch_count>("batch_count");
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

        service_status status = connector.reset_session(session_id);

        EXPECT_EQ(status.type, service_status_type::SUCCESS);
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

        service_status status = connector.reset_session(session_id);

        EXPECT_EQ(status.type, service_status_type::FAILURE);
    }
}

TEST(service_connector, close)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "SUCCESS";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id;
        EXPECT_CALL(*mock, del(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_status status = connector.close_session(session_id);

        EXPECT_EQ(status.type, service_status_type::SUCCESS);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "FAILURE";
        auto   expected_response        = http_response(http::status_ok, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id;
        EXPECT_CALL(*mock, del(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_status status = connector.close_session(session_id);

        EXPECT_EQ(status.type, service_status_type::FAILURE);
    }
}

TEST(service_connector, next)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        auto         buffer_map = make_shared<fixed_buffer_map>();
        stringstream serialized_buffer_map;
        buffer_map->serialize(serialized_buffer_map);
        std::vector<char> encoded_buffer_map = nervana::base64::encode(
            serialized_buffer_map.str().data(), serialized_buffer_map.str().size());

        auto expected_next_response = next_response(buffer_map);
        auto expected_response      = http_response(
            http::status_ok, string(encoded_buffer_map.begin(), encoded_buffer_map.end()));
        string endpoint = "/api/v1/dataset/" + session_id + "/next";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<next_response> response = connector.get_next(session_id);

        EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
        EXPECT_TRUE(response.data == expected_next_response);
    }

    // status type is not success
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        json expected_json;
        expected_json["status"]["type"] = "END_OF_DATASET";
        auto   expected_response        = http_response(http::status_no_data, expected_json.dump());
        string endpoint                 = "/api/v1/dataset/" + session_id + "/next";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        service_response<next_response> response = connector.get_next(session_id);

        EXPECT_EQ(response.status.type, service_status_type::END_OF_DATASET);
    }

    // next has improper value (is empty)
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        auto   expected_response = http_response(http::status_ok, "");
        string endpoint          = "/api/v1/dataset/" + session_id + "/next";
        EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

        ASSERT_THROW(connector.get_next(session_id), std::runtime_error);
    }
}

TEST(service_connector, names_and_shapes)
{
    // success scenario
    {
        auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
        service_connector connector(mock);

        vector<pair<string, shape_type>> nas = get_names_and_shapes();
        stringstream serialized_nas;
        serialized_nas << nas;

        json expected_json;
        expected_json["status"]["type"]           = "SUCCESS";
        expected_json["data"]["names_and_shapes"] = serialized_nas.str();
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
    template <property    P>
    service_response<int> get_property(service_connector& connector, const string& _session_id);

    template <>
    service_response<int> get_property<property::batch_count>(service_connector& connector,
                                                              const string&      _session_id)
    {
        return connector.get_batch_count(_session_id);
    }

    template <>
    service_response<int> get_property<property::batch_size>(service_connector& connector,
                                                             const string&      _session_id)
    {
        return connector.get_batch_size(_session_id);
    }

    template <>
    service_response<int> get_property<property::record_count>(service_connector& connector,
                                                               const string&      _session_id)
    {
        return connector.get_record_count(_session_id);
    }

    template <property P>
    void test_get_property(const string& property)
    {
        // success scenario
        {
            auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
            service_connector connector(mock);

            json expected_json;
            expected_json["status"]["type"] = "SUCCESS";
            expected_json["data"][property] = "64";
            auto   expected_response        = http_response(http::status_ok, expected_json.dump());
            string endpoint                 = "/api/v1/dataset/" + session_id + "/" + property;
            EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

            service_response<int> response = get_property<P>(connector, session_id);

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
            string endpoint                 = "/api/v1/dataset/" + session_id + "/" + property;
            EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

            service_response<int> response = get_property<P>(connector, session_id);

            EXPECT_EQ(response.status.type, service_status_type::FAILURE);
        }

        // property has improper value
        {
            auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
            service_connector connector(mock);

            json expected_json;
            expected_json["status"]["type"]        = "SUCCESS";
            expected_json["status"]["description"] = "some description";
            expected_json["data"][property]        = "";
            auto   expected_response = http_response(http::status_ok, expected_json.dump());
            string endpoint          = "/api/v1/dataset/" + session_id + "/" + property;
            EXPECT_CALL(*mock, get(endpoint, http_query_t())).WillOnce(Return(expected_response));

            ASSERT_THROW(get_property<P>(connector, session_id), std::runtime_error);
        }
    }
}
