#include "helpers.hpp"

#include "gtest/gtest.h"

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

void dump_vector_of_strings(vector<string>& words) {
    for(auto word = words.begin(); word != words.end(); ++word) {
        cout << *word << endl;
    }
}

void assert_vector_unique(vector<string>& words) {
    sort(words.begin(), words.end());
    for(auto word = words.begin(); word != words.end() - 1; ++word) {
        ASSERT_NE(*word, *(word + 1));
    }
}
