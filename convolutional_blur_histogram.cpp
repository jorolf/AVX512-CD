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
using ElementType = int32_t;
const size_t ELEMENT_COUNT = VECTOR_SIZE / sizeof(ElementType);


__m512i clamp(__m512i value, ElementType low, ElementType high) {
    __m512i high_vec = _mm512_set1_epi32(high);
    value = _mm512_min_epi32(high_vec, value);

    __m512i low_vec = _mm512_set1_epi32(low);
    value = _mm512_max_epi32(low_vec, value);

    return value;
}

template<typename type, size_t length>
void printArray(std::array<type, length> &array) {
    std::cout << "[";
    for (const auto &element: array) {
        std::cout << std::setw(8) << element;
    }
    std::cout << "]" << std::endl;
}

void printVector(__m512i vec) {
    ElementType typedVector[ELEMENT_COUNT];
    _mm512_storeu_si512(typedVector, vec);

    std::cout << "<";
    for (int i : typedVector) {
        std::cout << std::setw(8) << i;
    }
    std::cout << ">" << std::endl;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) convolutionalHistogramShiftedVector(std::vector<ElementType> &data, size_t rows, size_t cols) {
    std::array<ElementType, range> histogram = {};

    if (data.size() != rows * cols) return histogram;

    auto inc_const = _mm512_set1_epi32(1);
    auto div_multiplier = _mm512_set1_epi32(-477218588);
    __mmask16 div_mask = 0xAAAA;

    for (size_t current_row = 1; current_row < rows - 1; current_row++) {
        size_t current_col = 1;
        for (; current_col + ELEMENT_COUNT <= cols - 1; current_col += ELEMENT_COUNT) {

            auto topLeft = _mm512_loadu_si512(&data[(current_row + 1) * cols + (current_col + 1)]);
            auto topCenter = _mm512_loadu_si512(&data[(current_row + 1) * cols + current_col]);
            auto topRight = _mm512_loadu_si512(&data[(current_row + 1) * cols + (current_col - 1)]);
            auto middleLeft = _mm512_loadu_si512(&data[current_row * cols + (current_col + 1)]);

            auto middle = _mm512_loadu_si512(&data[current_row * cols + current_col]);

            auto middleRight = _mm512_loadu_si512(&data[current_row * cols + (current_col - 1)]);
            auto bottomLeft = _mm512_loadu_si512(&data[(current_row - 1) * cols + (current_col + 1)]);
            auto bottomCenter = _mm512_loadu_si512(&data[(current_row - 1) * cols + current_col]);
            auto bottomRight = _mm512_loadu_si512(&data[(current_row - 1) * cols + (current_col - 1)]);


            auto data_vec =
                    _mm512_add_epi32(
                            _mm512_add_epi32(
                                    _mm512_add_epi32(
                                            _mm512_add_epi32(
                                                    _mm512_add_epi32(
                                                            _mm512_add_epi32(
                                                                    _mm512_add_epi32(
                                                                            _mm512_add_epi32(middle, topLeft),
                                                                    topCenter),
                                                            topRight),
                                                    middleLeft),
                                            middleRight),
                                    bottomLeft),
                            bottomCenter),
                    bottomRight);

            // Adapted from http://www.agner.org/optimize/#vectorclass
            // and https://github.com/vectorclass/version2/blob/fee0601edd3c99845f4b7eeb697cff0385c686cb/vectori128.h#L6536
            auto lower = _mm512_mul_epi32(data_vec, div_multiplier);
            lower = _mm512_srli_epi64(lower, 32);
            auto upper = _mm512_srli_epi64(data_vec, 32);
            upper = _mm512_mul_epi32(upper, div_multiplier);
            data_vec = _mm512_add_epi32(_mm512_mask_blend_epi32(div_mask, lower, upper), data_vec);
            data_vec = _mm512_srai_epi32(data_vec, 3);

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

        for (; current_col < cols - 1; current_col++) {
            ElementType value = data[current_row * cols + current_col];
            value += data[(current_row + 1) * cols + (current_col + 1)];
            value += data[(current_row + 1) * cols + current_col];
            value += data[(current_row + 1) * cols + (current_col - 1)];
            value += data[current_row * cols + (current_col + 1)];
            value += data[current_row * cols + (current_col - 1)];
            value += data[(current_row - 1) * cols + (current_col + 1)];
            value += data[(current_row - 1) * cols + current_col];
            value += data[(current_row - 1) * cols + (current_col - 1)];

            value /= 9;

            histogram[value]++;
        }
    }

    return histogram;
}

template<ElementType range>
std::array<ElementType, range> __attribute__ ((noinline)) convolutionalHistogramSequential(std::vector<ElementType> &data, size_t rows, size_t cols) {
    std::array<ElementType, range> histogram = {};

    if (data.size() != rows * cols) return histogram;

    for (size_t current_row = 1; current_row < rows - 1; current_row++) {
        for (size_t current_col = 1; current_col < cols - 1; current_col++) {
            ElementType value = data[current_row * cols + current_col];
            value += data[(current_row + 1) * cols + (current_col + 1)];
            value += data[(current_row + 1) * cols + current_col];
            value += data[(current_row + 1) * cols + (current_col - 1)];
            value += data[current_row * cols + (current_col + 1)];
            value += data[current_row * cols + (current_col - 1)];
            value += data[(current_row - 1) * cols + (current_col + 1)];
            value += data[(current_row - 1) * cols + current_col];
            value += data[(current_row - 1) * cols + (current_col - 1)];

            value /= 9;

            histogram[value]++;
        }
    }

    return histogram;
}

template<ElementType histogram_range>
void testRange() {
    const size_t rows = 1000, cols = 1000;
    std::cout << "Histogram between 0 and " << histogram_range << " (exclusive):" << std::endl;

    std::random_device seed;
    std::default_random_engine rnd(seed());
    std::uniform_int_distribution<ElementType> dist(0, histogram_range);
    std::vector<ElementType> data(rows * cols);

    for (ElementType &i: data)
        i = dist(rnd);

    std::cout << "Sequential:" << std::endl;
    auto before = std::chrono::high_resolution_clock::now();
    auto result = convolutionalHistogramSequential<histogram_range>(data, rows, cols);
    auto after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Vector Instructions (shifted vector):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = convolutionalHistogramShiftedVector<histogram_range>(data, rows, cols);
    after = std::chrono::high_resolution_clock::now();
    printArray(result);
    std::cout << "The calculation took " << std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() << "µs" << std::endl;

    std::cout << "Sequential (again):" << std::endl;
    before = std::chrono::high_resolution_clock::now();
    result = convolutionalHistogramSequential<histogram_range>(data, rows, cols);
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
