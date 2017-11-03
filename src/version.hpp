/*
 * Copyright 2017 Intel(R) Nervana(TM)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if !defined(AEON_VERSION_HPP_INCLUDED_)
#define AEON_VERSION_HPP_INCLUDED_

#pragma once

#if !defined(BUILD_VERSION_MAJOR)
#define BUILD_VERSION_MAJOR 0
#endif

#if !defined(BUILD_VERSION_MINOR)
#define BUILD_VERSION_MINOR 0
#endif

#if !defined(BUILD_VERSION_PATCH)
#define BUILD_VERSION_PATCH 0
#endif

#if !defined(BUILD_NUMBER)
#define BUILD_NUMBER 0
#endif

namespace nervana
{
    namespace aeon
    {
        namespace version
        {
            constexpr int major{BUILD_VERSION_MAJOR};
            constexpr int minor{BUILD_VERSION_MINOR};
            constexpr int patch{BUILD_VERSION_PATCH};

            constexpr int abi{BUILD_VERSION_MAJOR};
        }

        namespace build
        {
            constexpr int number{BUILD_NUMBER};
#if defined(BUILD_EXTERNAL)
            constexpr bool internal{false};
#else  /* !BUILD_EXTERNAL */
            constexpr bool internal{true};
#endif /* BUILD_EXTERNAL */

#if defined(BUILD_EXPERIMENTAL)
            constexpr bool experimental{true};
#else  /* !BUILD_EXPERIMENTAL */
            constexpr bool experimental{false};
#endif /* BUILD_EXPERIMENTAL */
        }
    }
}

#endif /* AEON_VERSION_HPP_INCLUDED_ */
