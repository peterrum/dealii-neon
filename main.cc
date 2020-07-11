#include <iostream>

#include "vectorization.h"

int
main()
{
  using namespace dealii;

  VectorizedArray a(4.0);
  VectorizedArray b(4.0);

  std::cout << a[0] << std::endl;
  std::cout << a[1] << std::endl;
  std::cout << a[2] << std::endl;
  std::cout << a[3] << std::endl;

  a += b;

  a = std::sqrt(a);

  std::cout << a[0] << std::endl;

  return 0;
}