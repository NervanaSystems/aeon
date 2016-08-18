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

#include <unistd.h>

#include "gtest/gtest.h"

// cringe
#define private public
#include "block_loader_nds.hpp"

using namespace std;
using namespace nervana;

// NDSMockServer starts a python process in the constructor and kills the
// process in the destructor
class NDSMockServer {
public:
    NDSMockServer() {
        cout << "starting mock nds server ..." << endl;
        pid_t pid = fork();
        if(pid == 0) {
            int i = execl("../test/start_nds_server", (char*)0);
            if(i) {
                cout << "error starting nds_server: " << strerror(i) << endl;
            }
            exit(1);
        }

        // sleep for 3 seconds to let the nds_server come up
        usleep(3 * 1000 * 1000);
        _pid = pid;
    }

    ~NDSMockServer() {
        // kill the python process running the mock NDS
        kill(_pid, 15);
    }

private:
    pid_t _pid;
};

NDSMockServer s;

TEST(block_loader_nds, curl_stream) {
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    stringstream stream;
    client.get("http://127.0.0.1:5000/test_pattern/", stream);

    stringstream expected;
    for(int i = 0; i < 1024; ++i) {
        expected << "0123456789abcdef";
    }
    ASSERT_EQ(stream.str(), expected.str());
}

TEST(block_loader_nds, object_count) {
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    // 200 and 5 are hard coded in the mock nds server
    ASSERT_EQ(client.objectCount(), 200);
    ASSERT_EQ(client.blockCount(), 5);
}


TEST(block_loader_nds, cpio) {
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    buffer_in_array dest(2);
    ASSERT_EQ(dest.size(), 2);
    ASSERT_EQ(dest[0]->get_item_count(), 0);

    client.loadBlock(dest, 0);

    ASSERT_EQ(dest[0]->get_item_count(), 2);
}
