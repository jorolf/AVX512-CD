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
std::array<ElementType, range> __attribute__ ((noinline)) histogramComplete(std::vector<ElementType> &data) {
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
        histogram[data[i]]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramWithoutConflict(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_set4_epi32(1, 2, 4, 8);
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
        histogram[data[i]]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramWithoutGatherScatter(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    auto inc_const = _mm512_set1_epi32(1);

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto histogram_vec = _mm512_loadu_si512(histogram.data());
        histogram_vec = _mm512_add_epi32(histogram_vec, inc_const);

        while (_mm512_test_epi32_mask(conflicts, conflicts) != 0) {
            auto conflicts_bit1 = _mm512_and_si512(conflicts, inc_const);
            histogram_vec = _mm512_add_epi32(histogram_vec, conflicts_bit1);
            conflicts = _mm512_srli_epi32(conflicts, 1);
        }

        _mm512_storeu_si512(histogram.data(), histogram_vec);
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i]]++;
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) histogramWithoutConflictResolution(std::vector<ElementType> &data) {
    std::array<ElementType, range> histogram = {};

    for (int i = 0; i + ELEMENT_COUNT <= data.size(); i += ELEMENT_COUNT) {
        auto data_vec = _mm512_loadu_si512(data.data() + i);
        auto conflicts = _mm512_conflict_epi32(data_vec);
        auto histogram_vec = _mm512_i32gather_epi32(data_vec, histogram.data(), sizeof(ElementType));
        histogram_vec = _mm512_add_epi32(histogram_vec, conflicts);

        _mm512_i32scatter_epi32(histogram.data(), data_vec, histogram_vec, sizeof(ElementType));
    }

    for (size_t i = data.size() & ~(ELEMENT_COUNT - 1); i < data.size(); i++) {
        histogram[data[i]]++;
    }

    return histogram;
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

    std::cout << "Complete:" << std::endl;
    auto before = std::chrono::high_resolution_clock::now();
    histogramComplete<histogram_range>(data);
    auto after = std::chrono::high_resolution_clock::now();
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Without Conflict:" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    histogramWithoutConflict<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Without Scatter and Gather:" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    histogramWithoutGatherScatter<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Without Conflict Resolution:" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    histogramWithoutConflictResolution<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Complete (again):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    histogramComplete<histogram_range>(data);
    after = std::chrono::high_resolution_clock::now();
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

}

int main() {
    testRange<16>();
    std::cout << "\n----------------------------------------\n";
    testRange<64>();
    std::cout << "\n----------------------------------------\n";
    testRange<256>();

    return 0;
}
