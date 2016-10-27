#include <algorithm>
#include <fstream>
#include <dirent.h>

#include "helpers.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace nervana;

vector<string> buffer_to_vector_of_strings(buffer_in& b) {
    vector<string> words;
    for(auto i = 0; i != b.get_item_count(); ++i) {
        vector<char>& s = b.get_item(i);
        words.push_back(string(s.data(), s.size()));
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
