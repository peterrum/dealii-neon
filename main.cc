#include <iostream>

#include "vectorization.h"

using namespace dealii;

template <typename Number, std::size_t width>
void
test()
{
  VectorizedArray<Number, width> a(width);
  VectorizedArray<Number, width> b;

  for (unsigned int i = 0; i < width; ++i)
    b[i] = i + 2;

  std::cout << a << std::endl;
  std::cout << b << std::endl;

  a -= b;

  std::cout << a << std::endl;
  std::cout << std::abs(a) << std::endl;
  std::cout << std::sqrt(std::abs(a)) << std::endl;
  std::cout << std::min(a, b) << std::endl;
  std::cout << std::max(a, b) << std::endl;
}

int
main()
{
  test<float, 4>();

  return 0;
}