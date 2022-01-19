#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <bitset>

const size_t VECTOR_SIZE = 512 / 8;

template<typename ElementType>
void printVector(__m512i vec) {
    ElementType typedVector[VECTOR_SIZE / sizeof(ElementType)];
    _mm512_storeu_si512(typedVector, vec);

    std::cout << "<";
    for (auto i : typedVector) {
        std::cout << std::setw(sizeof(ElementType) * 2 + 1) << i;
    }
    std::cout << ">" << std::endl;
}

template<typename ElementType, size_t bitCount>
void printBitVector(__m512i vec) {
    ElementType typedVector[VECTOR_SIZE / sizeof(ElementType)];
    _mm512_storeu_si512(typedVector, vec);

    std::cout << "<";
    for (auto i : typedVector) {
        std::cout << std::setw(bitCount + 1) << std::bitset<bitCount>(i);
    }
    std::cout << ">" << std::endl;
}

void testMaskBroadcast() {
    __mmask8 mask8 = 0b10010110;
    std::cout << "Mask: " << +mask8 << std::endl;

    __m512i vector = _mm512_broadcastmb_epi64(mask8);

    std::cout << "Vector: ";
    printVector<int64_t>(vector);


    __mmask16 mask16 = 0xCAFE;
    std::cout << "Mask: " << +mask16 << std::endl;

    vector = _mm512_broadcastmw_epi32(mask16);

    std::cout << "Vector: ";
    printVector<int32_t>(vector);
}

void testZeroCount() {
    __m512i vector = _mm512_set_epi64(
            0xDECE'476C, 0x0B6F'C3FA'3E1F,
            0x7116,      0x5205'A0F2'E814'2F3A,
            0xCAFE,     -0x5205'A0F2'E814'2F3A,
            0x0,         0x0);
    std::cout << "Vector: ";
    printVector<int64_t>(vector);

    __m512i zeroVector = _mm512_lzcnt_epi64(vector);
    std::cout << "Zero bits in Vector: ";
    printVector<int64_t>(zeroVector);
}

void testConflictDetection() {
    __m512i vector = _mm512_set_epi64(
            0x1, 0x2,
            0x1, 0x4,
            0x1, 0x2,
            0x1, 0x8);
    std::cout << "Vector: ";
    printVector<int64_t>(vector);

    __m512i conflicts = _mm512_conflict_epi64(vector);
    std::cout << "Conflicts in Vector: ";
    printBitVector<int64_t, 8>(conflicts);
}

int main() {
    std::cout << std::hex;

    std::cout << "Broadcast: " << std::endl;
    testMaskBroadcast();
    std::cout << std::endl << "Zero Count: " << std::endl;
    testZeroCount();
    std::cout << std::endl << "Conflict Detection: " << std::endl;
    testConflictDetection();

    return 0;
}
