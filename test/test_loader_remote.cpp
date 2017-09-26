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
    MOCK_METHOD1(reset_session, service_status(const std::string& id));
    MOCK_METHOD1(close_session, service_status(const std::string& id));

    MOCK_METHOD1(get_next, service_response<next_response>(const std::string& id));
    MOCK_METHOD1(get_names_and_shapes, service_response<names_and_shapes>(const std::string& id));
    MOCK_METHOD1(get_record_count, service_response<int>(const std::string& id));
    MOCK_METHOD1(get_batch_size, service_response<int>(const std::string& id));
    MOCK_METHOD1(get_batch_count, service_response<int>(const std::string& id));
};

TEST(loader_remote, new_session_scenario)
{
    auto           mock         = make_shared<mock_service>();
    json           config       = {{"server", {"address", "localhost"}, {"port", 34568}}};
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
    EXPECT_CALL(*mock, create_session(config.dump())).WillOnce(Return(expected_create_session));
    EXPECT_CALL(*mock, get_names_and_shapes(session_id)).WillOnce(Return(expected_nas));
    EXPECT_CALL(*mock, get_record_count(session_id)).WillOnce(Return(expected_record_count));
    EXPECT_CALL(*mock, get_batch_size(session_id)).WillOnce(Return(expected_batch_size));
    EXPECT_CALL(*mock, get_batch_count(session_id)).WillOnce(Return(expected_batch_count));

    loader_remote loader(mock, config.dump());

    EXPECT_EQ(loader.get_names_and_shapes(), nas);
    EXPECT_EQ(loader.record_count(), record_count);
    EXPECT_EQ(loader.batch_size(), batch_size);
    EXPECT_EQ(loader.batch_count(), batch_count);
    EXPECT_EQ(loader.get_current_config(), config);

    // testing iteration
    {
        auto fbm = make_shared<nervana::fixed_buffer_map>();
        auto expected_batch = service_response<next_response>(status_success, next_response(fbm));
        auto expected_end_of_data = service_response<next_response>(
            service_status(service_status_type::END_OF_DATASET, ""), next_response());

        // iteration
        {
            testing::InSequence dummy;
            EXPECT_CALL(*mock, reset_session(session_id)).WillOnce(Return(status_success));
            EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_batch));
            EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_end_of_data));

            int index = 0;
            for (const auto& batch : loader)
            {
                if (index == 0)
                {
                    ostringstream batch_serialized, fbm_serialized;
                    batch.serialize(batch_serialized);
                    fbm->serialize(fbm_serialized);
                    EXPECT_EQ(batch_serialized.str(), fbm_serialized.str());
                }
                index++;
            }
        }

        // reset successful
        {
            // batch is retrieved, because iteration has finished in previous step
            EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_batch));
            loader.get_current_iter();

            // no batch retrieval
            loader.get_current_iter();
            EXPECT_EQ(loader.position(), 3);

            EXPECT_CALL(*mock, reset_session(session_id)).WillOnce(Return(status_success));
            loader.reset();

            // reset makes get_current_iter to retrieve data
            EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_batch));
            loader.get_current_iter();
            EXPECT_EQ(loader.position(), 0);
        }

        // reset unsuccessful
        {
            EXPECT_CALL(*mock, reset_session(session_id))
                .WillOnce(Return(service_status(service_status_type::FAILURE, "some message")));
            EXPECT_THROW(loader.reset(), runtime_error);
        }
    }
}

TEST(loader_remote, shared_session_scenario)
{
    string session_id = "SESSION_ID";
    auto   mock       = make_shared<mock_service>();
    json   config     = {
        {"server", {{"address", "localhost"}, {"port", 34568}, {"session_id", session_id}}}};
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
    EXPECT_CALL(*mock, get_names_and_shapes(session_id)).WillOnce(Return(expected_nas));
    EXPECT_CALL(*mock, get_record_count(session_id)).WillOnce(Return(expected_record_count));
    EXPECT_CALL(*mock, get_batch_size(session_id)).WillOnce(Return(expected_batch_size));
    EXPECT_CALL(*mock, get_batch_count(session_id)).WillOnce(Return(expected_batch_count));

    loader_remote loader(mock, config.dump());

    EXPECT_EQ(loader.get_names_and_shapes(), nas);
    EXPECT_EQ(loader.record_count(), record_count);
    EXPECT_EQ(loader.batch_size(), batch_size);
    EXPECT_EQ(loader.batch_count(), batch_count);
    EXPECT_EQ(loader.get_current_config(), config);

    // testing iteration
    {
        auto fbm = make_shared<nervana::fixed_buffer_map>();
        auto expected_batch = service_response<next_response>(status_success, next_response(fbm));
        auto expected_end_of_data = service_response<next_response>(
            service_status(service_status_type::END_OF_DATASET, ""), next_response());

        testing::InSequence dummy;
        EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_batch));
        EXPECT_CALL(*mock, get_next(session_id)).WillOnce(Return(expected_end_of_data));

        // iteration
        {
            int index = 0;
            for (const auto& batch : loader)
            {
                if (index == 0)
                {
                    ostringstream batch_serialized, fbm_serialized;
                    batch.serialize(batch_serialized);
                    fbm->serialize(fbm_serialized);
                    EXPECT_EQ(batch_serialized.str(), fbm_serialized.str());
                }
                index++;
            }
        }

        // reset does nothing in case of shared session
        {
            loader.reset();
        }
    }
}
