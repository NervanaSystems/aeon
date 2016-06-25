import _chromicorn
import numpy as np


class BBex(object):
    def consume(self, ary):
        self.ary = np.zeros_like(ary)
        self.ary[:] = ary.T

be = BBex()
_chromicorn.chromicorn(2, 2, be)

print be.ary
