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

#include <cstdint>
#include <immintrin.h>

namespace transpose
{
    namespace sse
    {
#define combine_4_2bits(n0, n1, n2, n3) (n0 + (n1 << 2) + (n2 << 4) + (n3 << 6))
#define _128_shuffle(x, y, n0, n1, n2, n3) _mm_shuffle_ps(x, y, combine_4_2bits(n0, n1, n2, n3))
#define _128i_shuffle(x, y, n0, n1, n2, n3)                                                        \
    _mm_castps_si128(_128_shuffle(_mm_castsi128_ps(x), _mm_castsi128_ps(y), n0, n1, n2, n3))

        inline void _128i_store(unsigned char* p, __m128i x)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(p), x);
        }

        inline __m128i _128i_load(const unsigned char* p)
        {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
        }

        inline __m128i transpose_4x4(__m128i m)
        {
            __m128i y = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

            return _mm_shuffle_epi8(m, y);
        }

        inline void transpose_4x4_dwords(__m128i  w0,
                                         __m128i  w1,
                                         __m128i  w2,
                                         __m128i  w3,
                                         __m128i& r0,
                                         __m128i& r1,
                                         __m128i& r2,
                                         __m128i& r3)
        {
            // 0  1  2  3
            // 4  5  6  7
            // 8  9  10 11
            // 12 13 14 15
            __m128i x0 = _128i_shuffle(w0, w1, 0, 1, 0, 1); // 0 1 4 5
            __m128i x1 = _128i_shuffle(w0, w1, 2, 3, 2, 3); // 2 3 6 7
            __m128i x2 = _128i_shuffle(w2, w3, 0, 1, 0, 1); // 8 9 12 13
            __m128i x3 = _128i_shuffle(w2, w3, 2, 3, 2, 3); // 10 11 14 15
            r0         = _128i_shuffle(x0, x2, 0, 2, 0, 2);
            r1         = _128i_shuffle(x0, x2, 1, 3, 1, 3);
            r2         = _128i_shuffle(x1, x3, 0, 2, 0, 2);
            r3         = _128i_shuffle(x1, x3, 1, 3, 1, 3);
        }

        inline void transpose_16x16(__m128i x[16])
        {
            __m128i w[4][4];
            transpose_4x4_dwords(x[0], x[1], x[2], x[3], w[0][0], w[0][1], w[0][2], w[0][3]);
            transpose_4x4_dwords(x[4], x[5], x[6], x[7], w[1][0], w[1][1], w[1][2], w[1][3]);
            transpose_4x4_dwords(x[8], x[9], x[10], x[11], w[2][0], w[2][1], w[2][2], w[2][3]);
            transpose_4x4_dwords(x[12], x[13], x[14], x[15], w[3][0], w[3][1], w[3][2], w[3][3]);

            for (int row = 0; row < 4; ++row)
                for (int col    = 0; col < 4; ++col)
                    w[row][col] = transpose_4x4(w[row][col]);

            transpose_4x4_dwords(w[0][0], w[1][0], w[2][0], w[3][0], x[0], x[1], x[2], x[3]);
            transpose_4x4_dwords(w[0][1], w[1][1], w[2][1], w[3][1], x[4], x[5], x[6], x[7]);
            transpose_4x4_dwords(w[0][2], w[1][2], w[2][2], w[3][2], x[8], x[9], x[10], x[11]);
            transpose_4x4_dwords(w[0][3], w[1][3], w[2][3], w[3][3], x[12], x[13], x[14], x[15]);
        }

        void transpose(uint8_t* dest, const uint8_t* src, int rows, int cols)
        {
            const int block_size = 16;
            __m128i   row[block_size];
            for (int cb = 0; cb < cols; cb += block_size)
            {
                for (int rb = 0; rb < rows; rb += block_size)
                {
                    const uint8_t* src_c = src + rb * cols + cb;
                    uint8_t*       dst_c = dest + cb * rows + rb;

                    for (int i = 0; i < block_size; i++, src_c += cols)
                        row[i] = _128i_load(src_c);

                    transpose_16x16(row);

                    for (int i = 0; i < block_size; i++, dst_c += rows)
                        _128i_store(dst_c, row[i]);
                }
            }
        }
    }
}
