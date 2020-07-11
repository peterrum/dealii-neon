float32x4_t
vdivq_f32(float32x4_t a, float32x4_t b)
{
  float32x4_t temp;

  temp[0] = a[0] / b[0];
  temp[1] = a[1] / b[1];
  temp[2] = a[2] / b[2];
  temp[3] = a[3] / b[3];

  return temp;
}

float64x2_t
vld1q_dup_f64(float64_t const *ptr)
{
  float64x2_t temp;

  temp[0] = ptr[0];
  temp[1] = ptr[0];

  return temp;
}

float64x2_t
vaddq_f64(float64x2_t a, float64x2_t b)
{
  float64x2_t temp;

  temp[0] = a[0] + b[0];
  temp[1] = a[1] + b[1];

  return temp;
}

float64x2_t
vsubq_f64(float64x2_t a, float64x2_t b)
{
  float64x2_t temp;

  temp[0] = a[0] - b[0];
  temp[1] = a[1] - b[1];

  return temp;
}

float64x2_t
vmulq_f64(float64x2_t a, float64x2_t b)
{
  float64x2_t temp;

  temp[0] = a[0] * b[0];
  temp[1] = a[1] * b[1];

  return temp;
}

float64x2_t
vdivq_f64(float64x2_t a, float64x2_t b)
{
  float64x2_t temp;

  temp[0] = a[0] / b[0];
  temp[1] = a[1] / b[1];

  return temp;
}

void
vst1q_f64(float64_t *ptr, float64x2_t val)
{
  ptr[0] = val[0];
  ptr[1] = val[1];
}
