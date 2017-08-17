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

class mock_service : public service
{
public:
    MOCK_METHOD2(create_session, service_response<string>(const std::string& config));
    MOCK_METHOD2(get_names_and_shapes, service_response<names_and_shapes>(const std::string& id));
    MOCK_METHOD2(next, service_response<next_response>(const std::string& id));
    MOCK_METHOD2(reset, service_response<next_response>(const std::string& id));

    MOCK_METHOD2(record_count, service_response<int>(const std::string& id));
    MOCK_METHOD2(batch_size, service_response<int>(const std::string& id));
    MOCK_METHOD2(batch_count, service_response<int>(const std::string& id));
};

//namespace
//{
//pair<loader_remote, shared_ptr<mock_service>> create_loader_remote_with_mock();
//}

TEST(loader_remote, basic_scenario)
{
    auto    mock = shared_ptr<mock_service>(new mock_service());
    service srvc(mock);

    string config = "{}";
    json   expected_json;
    expected_json["status"]["type"] = "SUCCESS";
    expected_json["data"]["id"]     = session_id;
    auto expected_response = service_response<names_and_shapes>(service_status_type::SUCCESS, nas);
    EXPECT_CALL(*mock, get_names_and_shapes(session_id))
        .WillOnce(Return(names_and_shapes_response));

    service_response<string> response = connector.create_session(config);

    EXPECT_EQ(response.status.type, service_status_type::SUCCESS);
    EXPECT_EQ(response.data, session_id);
}

//namespace
//{
//pair<loader_remote, shared_ptr<mock_service>> create_loader_remote_with_mock()
//{
//auto              mock = shared_ptr<mock_http_connector>(new mock_http_connector());
//service_connector connector(mock);

//EXPECT_CALL(
//}
//}

