#include "helpers.hpp"

#include <algorithm>

vector<string> buffer_to_vector_of_strings(Buffer& b) {
    vector<string> words;
    int len;
    for(auto i = 0; i != b.getItemCount(); ++i) {
        const char* s = b.getItem(i, len);
        words.push_back(string(s, s + len));
    }

    return words;
}

bool sorted(vector<string> words) {
    return std::is_sorted(words.begin(), words.end());
}
