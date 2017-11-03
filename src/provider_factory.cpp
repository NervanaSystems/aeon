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

#include "provider_factory.hpp"
#include "provider.hpp"
#include "log.hpp"
#include <sstream>

using namespace std;

shared_ptr<nervana::provider_interface> nervana::provider_factory::create(nlohmann::json configJs)
{
    shared_ptr<nervana::provider_interface> rc;

    provider_config cc{configJs};

    for (auto j : cc.augmentation)
    {
        auto val = j.find("type");
        if (val == j.end())
        {
            throw std::invalid_argument("augmentation missing 'type'");
        }
    }

    nlohmann::json aug_config;
    if (cc.augmentation.size() > 0)
    {
        aug_config = cc.augmentation[0];
    }
    // else
    // {
    //     aug_config = nlohmann::json::object();
    // }
    rc = make_shared<provider::provider_base>(configJs, cc.etl, aug_config);

    return rc;
}

shared_ptr<nervana::provider_interface>
    nervana::provider_factory::clone(const shared_ptr<nervana::provider_interface>& r)
{
    return create(r->get_config());
}
