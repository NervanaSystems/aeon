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
#pragma once
#include <Python.h>
#include <iostream>
#include <algorithm>

namespace nervana
{
    namespace python
    {
        class ensure_gil
        {
        public:
            ensure_gil()
                : _state{PyGILState_Ensure()}
            {
            }

            ~ensure_gil() { PyGILState_Release(_state); }
        private:
            PyGILState_STATE _state;
        };

        class static_initialization
        {
        public:
            static static_initialization& Instance()
            {
                static static_initialization obj;
                return obj;
            }

        private:
            static_initialization();
            ~static_initialization();
            PyThreadState* _save = nullptr;
        };

#ifdef PYTHON_PLUGIN
        struct block_threads;
        struct allow_threads
        {
            allow_threads();
            ~allow_threads();

        private:
            friend struct block_threads;
            PyThreadState* _state{nullptr};
        };

        struct block_threads
        {
            block_threads(allow_threads& a);
            block_threads() = delete;
            ~block_threads();

        private:
            allow_threads& _parent;
            PyThreadState* _state{nullptr};
        };
#else
        struct allow_threads
        {
        };

        struct block_threads
        {
            block_threads(allow_threads&) {}
        };
#endif
    }
}
