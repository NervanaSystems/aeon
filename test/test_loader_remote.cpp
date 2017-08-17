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
#include "helpers.hpp"
#include "loader_remote.hpp"
#include "log.hpp"
#include "service.hpp"

using namespace std;
using namespace nervana;

using ::testing::Return;
using nlohmann::json;

class mock_service : public service
{
public:
    MOCK_METHOD1(create_session, service_response<string>(const std::string& config));
    MOCK_METHOD1(get_names_and_shapes, service_response<names_and_shapes>(const std::string& id));
    MOCK_METHOD1(next, service_response<next_response>(const std::string& id));
    MOCK_METHOD1(reset, service_status(const std::string& id));
    MOCK_METHOD1(close_session, service_status(const std::string& id));

    MOCK_METHOD1(record_count, service_response<int>(const std::string& id));
    MOCK_METHOD1(batch_size, service_response<int>(const std::string& id));
    MOCK_METHOD1(batch_count, service_response<int>(const std::string& id));
};

//namespace
//{
//pair<loader_remote, shared_ptr<mock_service>> create_loader_remote_with_mock();
//}

TEST(loader_remote, basic_scenario)
{
    auto           mock         = make_shared<mock_service>();
    string         config       = "{}";
    string         session_id   = "123";
    auto           nas          = get_names_and_shapes();
    int            batch_size   = 64;
    int            batch_count  = 3;
    int            record_count = batch_size * batch_count;
    service_status status_success(service_status_type::SUCCESS, "");

    auto expected_create_session = service_response<string>(status_success, session_id);
    auto expected_nas            = service_response<names_and_shapes>(status_success, nas);
    auto expected_record_count   = service_response<int>(status_success, record_count);
    auto expected_batch_size     = service_response<int>(status_success, batch_size);
    auto expected_batch_count    = service_response<int>(status_success, batch_count);
    EXPECT_CALL(*mock, create_session(config)).WillOnce(Return(expected_create_session));
    EXPECT_CALL(*mock, get_names_and_shapes(session_id)).WillOnce(Return(expected_nas));
    EXPECT_CALL(*mock, record_count(session_id)).WillOnce(Return(expected_record_count));
    EXPECT_CALL(*mock, batch_size(session_id)).WillOnce(Return(expected_batch_size));
    EXPECT_CALL(*mock, batch_count(session_id)).WillOnce(Return(expected_batch_count));

    loader_remote loader(mock, config);

    EXPECT_EQ(loader.get_names_and_shapes(), nas);
    EXPECT_EQ(loader.record_count(), record_count);
    EXPECT_EQ(loader.batch_size(), batch_size);
    EXPECT_EQ(loader.batch_count(), batch_count);


    // iteration
    {

        int index = 0;
        for(const auto& batch : loader)
        {
            index++;
        }

    }
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
