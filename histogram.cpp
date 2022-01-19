#include <iostream>
#include <array>
#include <vector>
#include <cstdint>
#include <ctime>
#include <random>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <algorithm>

const size_t VECTOR_SIZE = 512 / 8;
using ElementType = uint32_t;
const size_t ELEMENT_COUNT = VECTOR_SIZE / sizeof(ElementType);

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramMaskedScatter(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);
    auto and_const = _mm512_set1_epi32(-1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto gathered_histogram = _mm512_i32gather_epi32(data_vec, histogram.data(), sizeof(ElementType));

        auto conflict_mask = _mm512_testn_epi32_mask(conflicts, and_const);
        auto incremented_histogram = _mm512_add_epi32(gathered_histogram, inc_const);
        _mm512_mask_i32scatter_epi32(histogram.data(), conflict_mask, data_vec, incremented_histogram, sizeof(ElementType));

        for(uint16_t mask = ~conflict_mask, j = 0; mask != 0; mask >>= 1) {
            if ((mask & 1) != 0) {
                histogram[data[j + i]]++;
            }
            j++;
        }
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i], 0u, range]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramShiftedMask(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto gathered_histogram = _mm512_i32gather_epi32(data_vec, histogram.data(), sizeof(ElementType));

        __mmask16 conflict_mask = UINT16_MAX;
        bool conflicts_left = true;

        while (conflicts_left) {
            gathered_histogram = _mm512_mask_add_epi32(gathered_histogram, conflict_mask, gathered_histogram, inc_const);
            conflict_mask = _mm512_test_epi32_mask(conflicts, inc_const);
            conflicts_left = _mm512_test_epi32_mask(conflicts, conflicts) != 0;
            conflicts = _mm512_srli_epi32(conflicts, 1);
        }

        _mm512_i32scatter_epi32(histogram.data(), data_vec, gathered_histogram, sizeof(ElementType));
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i], 0u, range]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramShiftedVector(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto histogram_vec = _mm512_i32gather_epi32(data_vec, histogram.data(), sizeof(ElementType));
        histogram_vec = _mm512_add_epi32(histogram_vec, inc_const);

        while (_mm512_test_epi32_mask(conflicts, conflicts) != 0) {
            auto conflicts_bit1 = _mm512_and_si512(conflicts, inc_const);
            histogram_vec = _mm512_add_epi32(histogram_vec, conflicts_bit1);
            conflicts = _mm512_srli_epi32(conflicts, 1);
        }

        _mm512_i32scatter_epi32(histogram.data(), data_vec, histogram_vec, sizeof(ElementType));
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i], 0u, range]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramPOPCNT(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto histogram_vec = _mm512_i32gather_epi32(data_vec, histogram.data(), sizeof(ElementType));
        histogram_vec = _mm512_add_epi32(histogram_vec, inc_const);

        histogram_vec = _mm512_add_epi32(histogram_vec, _mm512_popcnt_epi32(conflicts));

        _mm512_i32scatter_epi32(histogram.data(), data_vec, histogram_vec, sizeof(ElementType));
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i], 0u, range]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramSequential(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    for (const auto &datum : data) {
        histogram[datum, 0u, range]++;
    }

    return histogram;
}

template<typename type, size_t length>
void printArray(std::array<type, length> &array) {
    std::cout << "[";
    for (const auto &element : array) {
        std::cout << std::setw(8) << element;
    }
    std::cout << "]" << std::endl;
}

template<ElementType histogram_range>
void testRange() {
    std::cout << "Histogram between 0 and " << histogram_range << " (exclusive):" << std::endl;

    std::random_device seed;
    std::default_random_engine rnd(seed());
    std::uniform_int_distribution<uint32_t> dist(0, histogram_range - 1);
    std::vector<uint32_t> data(10'000'000);

    for (uint32_t &i : data)
        i = dist(rnd);

    std::cout << "Sequential:" << std::endl;
    auto before = std::chrono::high_resolution_clock::now();
    auto result = histogramSequential<histogram_range>(data);
    auto after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Vector Instructions (masked scatter):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = histogramMaskedScatter<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Vector Instructions (shifted mask):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = histogramShiftedMask<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Vector Instructions (shifted vector):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = histogramShiftedVector<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

#ifdef __AVX512VPOPCNTDQ__
    std::cout << "Vector Instructions (popcnt intrinsic):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = histogramPOPCNT<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;
#endif

    std::cout << "Sequential (again):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = histogramSequential<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

}

int main() {
    testRange<8>();
    std::cout << "\n----------------------------------------\n";
    testRange<32>();
    std::cout << "\n----------------------------------------\n";
    testRange<256>();

    return 0;
}
