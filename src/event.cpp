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

#include "event.hpp"

nervana::event::event()
    : _ready(false)
{
}

void nervana::event::wait()
{
    std::unique_lock<std::mutex> lk(_mutex);
    while (!_ready)
        _condition.wait(lk);
    _ready = false;
}

void nervana::event::wait_multiple()
{
    std::unique_lock<std::mutex> lk(_mutex);
    _condition.wait(lk);
}

void nervana::event::notify()
{
    std::unique_lock<std::mutex> lk(_mutex);
    _ready = true;
    _condition.notify_one();
}

void nervana::event::notify_all()
{
    std::unique_lock<std::mutex> lk(_mutex);
    _ready = true;
    _condition.notify_all();
}
