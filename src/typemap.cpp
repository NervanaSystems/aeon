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

#include <tuple>

#include "typemap.hpp"

const std::string nervana::output_type::m_tp_name_json_name = "name";
const std::string nervana::output_type::m_np_type_json_name = "np_type";
const std::string nervana::output_type::m_cv_type_json_name = "cv_type";
const std::string nervana::output_type::m_size_json_name    = "size";

const std::string nervana::shape_type::m_shape_json_name     = "shape";
const std::string nervana::shape_type::m_otype_json_name     = "otype";
const std::string nervana::shape_type::m_byte_size_json_name = "byte_size";
const std::string nervana::shape_type::m_names_json_name     = "names";

using namespace nervana;
using nlohmann::json;

std::ostream& shape_type::serialize(std::ostream& out) const
{
    nlohmann::json json_out;
    to_json(json_out, *this);
    out << json_out;
    return out;
}

std::istream& shape_type::deserialize(std::istream& in)
{
    nlohmann::json json_in;
    in >> json_in;
    from_json(json_in, *this);
    return in;
}

std::ostream& operator<<(std::ostream& out, const nervana::shape_type& obj)
{
    return obj.serialize(out);
}

std::istream& operator>>(std::istream& in, nervana::shape_type& obj)
{
    return obj.deserialize(in);
}

std::ostream& operator<<(std::ostream& out,
                         const std::vector<std::pair<std::string, nervana::shape_type>>& obj)
{
    json json_out;
    for (auto el : obj)
        to_json(json_out[std::get<0>(el)], std::get<1>(el));

    out << json_out;
    return out;
}

std::istream& operator>>(std::istream& in,
                         std::vector<std::pair<std::string, nervana::shape_type>>& obj)
{
    json json_in;
    in >> json_in;

    for (json::iterator it = json_in.begin(); it != json_in.end(); ++it)
    {
        shape_type new_object;
        from_json(it.value(), new_object);
        obj.push_back(std::pair<std::string, nervana::shape_type>(it.key(), new_object));
    }

    return in;
}
