#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NEON_2_SSE.h"
#pragma GCC diagnostic pop

namespace dealii
{
  class VectorizedArray;
}

namespace std
{
  inline dealii::VectorizedArray
  sqrt(const dealii::VectorizedArray &x);
}

namespace dealii
{
  class VectorizedArray
  {
  public:
    using value_type = float;

    static const unsigned int n_array_elements = 4;

    VectorizedArray() = default;

    VectorizedArray(const double scalar)
    {
      this->operator=(scalar);
    }

    VectorizedArray &
    operator=(const double x)
    {
      data = vld1q_dup_f32(&x);
      return *this;
    }

    value_type &operator[](const unsigned int comp)
    {
      return *(reinterpret_cast<value_type *>(&data) + comp);
    }

    VectorizedArray &
    operator+=(const VectorizedArray &vec)
    {
      data = vaddq_f32(data, vec.data);
      return *this;
    }

    VectorizedArray &
    operator-=(const VectorizedArray &vec)
    {
      data = vsubq_f32(data, vec.data);
      return *this;
    }

    VectorizedArray &
    operator*=(const VectorizedArray &vec)
    {
      data = vmulq_f32(data, vec.data);
      return *this;
    }

#if false
  VectorizedArray &
  operator/=(const VectorizedArray &vec)
  {
    data = vdivq_f32(data, vec.data);
    return *this;
  }
#endif

    void
    load(const value_type *ptr)
    {
      data = vld1q_f32(ptr);
    }

    void
    store(value_type *ptr) const
    {
      vst1q_f32(ptr, data);
    }

    void
    gather(const value_type *base_ptr, const unsigned int *offsets)
    {
      for (unsigned int i = 0; i < n_array_elements; ++i)
        *(reinterpret_cast<value_type *>(&data) + i) = base_ptr[offsets[i]];
    }

    void
    scatter(const unsigned int *offsets, value_type *base_ptr) const
    {
      for (unsigned int i = 0; i < n_array_elements; ++i)
        base_ptr[offsets[i]] =
          *(reinterpret_cast<const value_type *>(&data) + i);
    }

    mutable float32x4_t data;

  private:
    VectorizedArray
    get_sqrt() const
    {
      VectorizedArray res;
      res.data = vsqrtq_f32(data);
      return res;
    }

    VectorizedArray
    get_abs() const
    {
      VectorizedArray res;
      res.data = vabsq_f32(data);
      return res;
    }

    VectorizedArray
    get_max(const VectorizedArray &other) const
    {
      VectorizedArray res;
      res.data = vmaxq_f32(data, other.data);
      return res;
    }

    VectorizedArray
    get_min(const VectorizedArray &other) const
    {
      VectorizedArray res;
      res.data = vminq_f32(data, other.data);
      return res;
    }

    friend VectorizedArray
    std::sqrt(const VectorizedArray &);
  };

} // namespace dealii

namespace std
{
  inline dealii::VectorizedArray
  sqrt(const dealii::VectorizedArray &x)
  {
    return x.get_sqrt();
  }

} // namespace std
