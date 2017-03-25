#include <immintrin.h>

#include <iostream>

#include <chrono>


 //! RapidJson implementation
 inline const char *SkipWhitespace_SIMD(const char* p) {
         // Fast return for single non-whitespace
         if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
                 ++p;
         else
             return p;

         // 16-byte align to the next boundary
         const char* nextAligned = reinterpret_cast<const char*>((reinterpret_cast<size_t>(p) + 15) & static_cast<size_t>(~15));
         while (p != nextAligned)
                 if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
                     ++p;
             else
                 return p;

         // The rest of string
         #define C16(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c }
         static const char whitespaces[4][16] = { C16(' '), C16('\n'), C16('\r'), C16('\t') };
         #undef C16

         const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&whitespaces[0][0]));
         const __m128i w1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&whitespaces[1][0]));
         const __m128i w2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&whitespaces[2][0]));
         const __m128i w3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&whitespaces[3][0]));

         for (;; p += 16) {
                 const __m128i s = _mm_load_si128(reinterpret_cast<const __m128i *>(p));
                 __m128i x = _mm_cmpeq_epi8(s, w0);
                 x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w1));
                 x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w2));
                 x = _mm_or_si128(x, _mm_cmpeq_epi8(s, w3));
                 unsigned short r = static_cast<unsigned short>(~_mm_movemask_epi8(x));
                 if (r != 0) {   // some of characters may be non-whitespace
                         return p + __builtin_ffs(r) - 1;
                     }
             }
     }

//! Naive implementation
inline const char * stupid_function(const char* p, std::string& str) {
    int i = 0;
    for (i = 0; i < str.length(); i = i + 1)
        if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
            ++p;
        else {
            return p;
        }
    return p;
}

//! AVX2 implementation - basically the rapidjson logic with 256 bit operations
inline const char * avx2_skipwhitespace(const char* p) {

    if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
        ++p;
    else {
        return p;
    }

// 16-byte align to the next boundary
    const char* nextAligned = reinterpret_cast<const char*>((reinterpret_cast<size_t>(p) + 31) & static_cast<size_t>(~31));
    while (p != nextAligned)
        if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
            ++p;
        else
            return p;


    // The rest of string
    #define C32(c) { c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c,c }
    static const char whitespaces[4][32] = { C32(' '), C32('\n'), C32('\r'), C32('\t') };
    #undef C32

    const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&whitespaces[0][0]));
    const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&whitespaces[1][0]));
    const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&whitespaces[2][0]));
    const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&whitespaces[3][0]));

    for (;; p += 32) {
        const __m256i s = _mm256_load_si256(reinterpret_cast<const __m256i *>(p));
        __m256i x = _mm256_cmpeq_epi8(s, w0);
        x = _mm256_or_si256(x, _mm256_cmpeq_epi8(s, w1));
        x = _mm256_or_si256(x, _mm256_cmpeq_epi8(s, w2));
        x = _mm256_or_si256(x, _mm256_cmpeq_epi8(s, w3));
        unsigned int r = static_cast<unsigned int>(~_mm256_movemask_epi8(x));
        if (r != 0) {   // some of characters may be non-whitespace
            return p + __builtin_ffs(r) - 1;
        }
    }
}

int main() {
    std::string str = "        bbbbb";
    std::cout<<"No avx: \n";

    //when running O3 compiler will very aggressively optimize the naive implementation, making tests useless
    #define NUM_OF_LOOPS 1

    auto p = str.c_str();
    const char* p2;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_OF_LOOPS; i++) {
        stupid_function(p, str);
        //    return SkipWhitespace(p, end);
    }
    p2 = stupid_function(p, str);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(t2.time_since_epoch() - t1.time_since_epoch()).count();
    std::cout << "First non-whitespace: " << p2 - p << " time: " << diff << "\n";

    std::cout<<"RapidJson sse: \n";
    t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_OF_LOOPS; i++) {
        SkipWhitespace_SIMD(p);
    }
    p2 = SkipWhitespace_SIMD(p);
    t2 = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::nanoseconds>(t2.time_since_epoch() - t1.time_since_epoch()).count();
    std::cout << "First non-whitespace: " << p2 - p << " time: " << diff << "\n";


    std::cout<<"With avx: \n";
    t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_OF_LOOPS; i++) {
        avx2_skipwhitespace(p);
        //    return SkipWhitespace(p, end);
    }
    p2 = avx2_skipwhitespace(p);
    t2 = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::nanoseconds>(t2.time_since_epoch() - t1.time_since_epoch()).count();
    std::cout << "First non-whitespace: " << p2 - p << " time: " << diff << "\n";

    return 0;
}