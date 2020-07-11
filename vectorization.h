#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "NEON_2_SSE.h"
#pragma GCC diagnostic pop

namespace dealii
{
  template <typename Number, std::size_t width>
  class VectorizedArray;
}

namespace std
{
  template <typename Number, std::size_t width>
  inline dealii::VectorizedArray<Number, width>
  sqrt(const dealii::VectorizedArray<Number, width> &x);
}

namespace dealii
{
  template <>
  class VectorizedArray<float, 4>
  {
  public:
    using value_type = float;

    static const unsigned int n_array_elements = 4;

    VectorizedArray() = default;

    VectorizedArray(const double scalar)
    {
      this->operator=(scalar);
    }

    VectorizedArray<value_type, n_array_elements> &
    operator=(const double x)
    {
      data = vld1q_dup_f32(&x);
      return *this;
    }

    value_type &operator[](const unsigned int comp)
    {
      return *(reinterpret_cast<value_type *>(&data) + comp);
    }

    const value_type &operator[](const unsigned int comp) const
    {
      return *(reinterpret_cast<const value_type *>(&data) + comp);
    }

    VectorizedArray<value_type, n_array_elements> &
    operator+=(const VectorizedArray<value_type, n_array_elements> &vec)
    {
      data = vaddq_f32(data, vec.data);
      return *this;
    }

    VectorizedArray<value_type, n_array_elements> &
    operator-=(const VectorizedArray<value_type, n_array_elements> &vec)
    {
      data = vsubq_f32(data, vec.data);
      return *this;
    }

    VectorizedArray<value_type, n_array_elements> &
    operator*=(const VectorizedArray<value_type, n_array_elements> &vec)
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
    VectorizedArray<value_type, n_array_elements>
    get_sqrt() const
    {
      VectorizedArray<value_type, n_array_elements> res;
      res.data = vsqrtq_f32(data);
      return res;
    }

    VectorizedArray<value_type, n_array_elements>
    get_abs() const
    {
      VectorizedArray<value_type, n_array_elements> res;
      res.data = vabsq_f32(data);
      return res;
    }

    VectorizedArray<value_type, n_array_elements>
    get_max(const VectorizedArray<value_type, n_array_elements> &other) const
    {
      VectorizedArray<value_type, n_array_elements> res;
      res.data = vmaxq_f32(data, other.data);
      return res;
    }

    VectorizedArray<value_type, n_array_elements>
    get_min(const VectorizedArray<value_type, n_array_elements> &other) const
    {
      VectorizedArray<value_type, n_array_elements> res;
      res.data = vminq_f32(data, other.data);
      return res;
    }

    template <typename Number2, std::size_t width2>
    friend VectorizedArray<Number2, width2>
    std::sqrt(const VectorizedArray<Number2, width2> &);

    template <typename Number2, std::size_t width2>
    friend VectorizedArray<Number2, width2>
    std::abs(const VectorizedArray<Number2, width2> &);

    template <typename Number2, std::size_t width2>
    friend VectorizedArray<Number2, width2>
    std::max(const VectorizedArray<Number2, width2> &,
             const VectorizedArray<Number2, width2> &);

    template <typename Number2, std::size_t width2>
    friend VectorizedArray<Number2, width2>
    std::min(const VectorizedArray<Number2, width2> &,
             const VectorizedArray<Number2, width2> &);
  };

  template <typename Number, std::size_t width>
  inline std::ostream &
  operator<<(std::ostream &out, const VectorizedArray<Number, width> &p)
  {
    for (unsigned int i = 0; i < width - 1; ++i)
      out << p[i] << ' ';
    out << p[width - 1];

    return out;
  }

} // namespace dealii

namespace std
{
  template <typename Number, std::size_t width>
  inline dealii::VectorizedArray<Number, width>
  sqrt(const dealii::VectorizedArray<Number, width> &x)
  {
    return x.get_sqrt();
  }

  template <typename Number, std::size_t width>
  inline dealii::VectorizedArray<Number, width>
  abs(const dealii::VectorizedArray<Number, width> &x)
  {
    return x.get_abs();
  }

  template <typename Number, std::size_t width>
  inline dealii::VectorizedArray<Number, width>
  max(const dealii::VectorizedArray<Number, width> &x,
      const dealii::VectorizedArray<Number, width> &y)
  {
    return x.get_max(y);
  }

  template <typename Number, std::size_t width>
  inline dealii::VectorizedArray<Number, width>
  min(const dealii::VectorizedArray<Number, width> &x,
      const dealii::VectorizedArray<Number, width> &y)
  {
    return x.get_min(y);
  }


} // namespace std
