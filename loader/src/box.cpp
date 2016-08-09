#include "box.hpp"

using namespace std;

ostream& operator<<(ostream& out, const nervana::box& b) {
    out << "(" << b.xmin << "," << b.ymin << ")(" << b.xmax << "," << b.ymax << ")";
    return out;
}

ostream& operator<<(ostream& out, const vector<nervana::box>& list) {
    for( const nervana::box& b : list ) {
        out << b << "\n";
    }
    return out;
}
