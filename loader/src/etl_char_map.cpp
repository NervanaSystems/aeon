#include "etl_char_map.hpp"

using namespace std;
using namespace nervana;

std::shared_ptr<char_map::decoded> char_map::extractor::extract(const char* in_array, int in_sz)
{
    uint32_t nvalid = std::min((uint32_t) in_sz, _max_length);
    string transcript(in_array, nvalid);
    vector<uint8_t> char_ints((vector<uint8_t>::size_type) _max_length, (uint8_t) 0);

    for (uint i=0; i<nvalid; i++)
    {
        auto l = _cmap.find(std::toupper(transcript[i]));
        uint8_t v = (l != _cmap.end()) ? l->second : UINT8_MAX;
        char_ints[i] = v;
    }
    auto rc = make_shared<char_map::decoded>(char_ints, nvalid);
    return rc;
}


void char_map::loader::load(char* out_ptr, std::shared_ptr<char_map::decoded> dc)
{
    for (auto c: dc->get_data())
    {
        *(out_ptr++) = c;
    }
}
