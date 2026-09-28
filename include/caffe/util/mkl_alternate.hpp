#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif  // USE_ACCELERATE

#include <math.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const float b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

// vPowx is spelled out instead of using DEFINE_VSL_UNARY_FUNC_WITH_PARAM so
// that the float specialization below can be declared before the vsPowx/vdPowx
// wrappers, which lets those wrappers pick up the RVV fast path. The generic
// template keeps the original element-wise libm powf semantics and CHECKs.
template<typename Dtype>
void vPowx(const int n, const Dtype* a, const Dtype b, Dtype* y) {
  CHECK_GT(n, 0); CHECK(a); CHECK(y);
  for (int i = 0; i < n; ++i) { y[i] = pow(a[i], b); }
}

#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
// Applies `fn` to every float lane of a[0..n) and stores into y[0..n),
// using a fixed VL (LMUL=m4) main loop plus a runtime-VL tail. Reading and
// writing the same chunk keeps in-place calls (a == y) safe.
template <typename Fn>
static inline void rvv_map_f32(const int n, const float* a, float* y, Fn fn) {
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  int i = 0;
  for (; i + static_cast<int>(vlmax) <= n; i += static_cast<int>(vlmax)) {
    const vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vlmax);
    __riscv_vse32_v_f32m4(y + i, fn(va, vlmax), vlmax);
  }
  if (i < n) {
    const size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(n - i));
    const vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
    __riscv_vse32_v_f32m4(y + i, fn(va, vl), vl);
  }
}

// The RVV path below evaluates exp2(b * log2(x)) with f64 vector
// intermediates. A pure f32 polynomial is not accurate enough (measured up
// to ~100 ULP vs libm powf), while the f64 pipeline yields a result within
// 1 ULP of libm powf on the target riscv64 libm (measured over the whole
// normal positive domain for |b| <= 1, including the bvlc_alexnet LRN
// exponent b = -0.75). The vector path is only used when the domain is
// provably safe: every input must be a positive normal float (so the
// exponent/mantissa decomposition is exact) and |b| <= 1 (so the result
// exponent stays inside the f64 normal range). Anything else, including
// negative / zero / subnormal / inf / NaN inputs and larger exponents, keeps
// the original per-element libm powf loop, so special values and their
// semantics are untouched.
static inline bool rvv_powx_domain_ok(const float* a, const int n,
                                      const float b) {
  if (!(b >= -1.0f && b <= 1.0f)) {
    return false;
  }
  for (int i = 0; i < n; ++i) {
    uint32_t u;
    std::memcpy(&u, &a[i], sizeof(u));
    const uint32_t ef = (u >> 23) & 0xFFu;
    if (ef == 0 || ef == 0xFF) {
      return false;
    }
    if (u >> 31) {
      return false;
    }
  }
  return true;
}

static inline void rvv_powx_pos_chunk(const float* a, const int i,
                                      const int n, const double b, float* y) {
  const size_t vl = __riscv_vsetvl_e32m2(static_cast<size_t>(n - i));
  const vfloat32m2_t xf = __riscv_vle32_v_f32m2(a + i, vl);
  const vfloat64m4_t x = __riscv_vfwcvt_f_f_v_f64m4(xf, vl);
  const vuint64m4_t xi = __riscv_vreinterpret_v_f64m4_u64m4(x);
  const vuint64m4_t ef = __riscv_vand_vx_u64m4(
      __riscv_vsrl_vx_u64m4(xi, 52, vl), 0x7FFull, vl);
  const vint64m4_t ei = __riscv_vsub_vx_i64m4(
      __riscv_vreinterpret_v_u64m4_i64m4(ef), 1023, vl);
  const vuint64m4_t mant = __riscv_vor_vx_u64m4(
      __riscv_vand_vx_u64m4(xi, 0xFFFFFFFFFFFFFull, vl),
      0x3FF0000000000000ull, vl);
  const vfloat64m4_t m = __riscv_vreinterpret_v_u64m4_f64m4(mant);
  const vfloat64m4_t t = __riscv_vfdiv_vv_f64m4(
      __riscv_vfsub_vf_f64m4(m, 1.0, vl),
      __riscv_vfadd_vf_f64m4(m, 1.0, vl), vl);
  const vfloat64m4_t t2 = __riscv_vfmul_vv_f64m4(t, t, vl);
  // ln(m) = 2 * t * (1 + t^2/3 + t^4/5 + ... + t^16/17), t in [0, 1/3)
  vfloat64m4_t p = __riscv_vfmv_v_f_f64m4(1.0 / 17.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 15.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 13.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 11.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 9.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 7.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 5.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0 / 3.0, vl);
  p = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(t2, p, vl), 1.0, vl);
  const vfloat64m4_t lnm = __riscv_vfmul_vf_f64m4(
      __riscv_vfmul_vv_f64m4(t, p, vl), 2.0, vl);
  const vfloat64m4_t log2x = __riscv_vfadd_vv_f64m4(
      __riscv_vfcvt_f_x_v_f64m4(ei, vl),
      __riscv_vfmul_vf_f64m4(lnm, 1.4426950408889634, vl), vl);
  const vfloat64m4_t yy = __riscv_vfmul_vf_f64m4(log2x, b, vl);
  const vint64m4_t ki = __riscv_vfcvt_x_f_v_i64m4(yy, vl);
  const vfloat64m4_t kf = __riscv_vfcvt_f_x_v_f64m4(ki, vl);
  const vfloat64m4_t f = __riscv_vfsub_vv_f64m4(yy, kf, vl);
  const vfloat64m4_t z = __riscv_vfmul_vf_f64m4(f, 0.6931471805599453, vl);
  // exp(z) = 1 + z + z^2/2! + ... + z^10/10!
  vfloat64m4_t ez = __riscv_vfmv_v_f_f64m4(1.0 / 3628800.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 362880.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 40320.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 5040.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 720.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 120.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 24.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0 / 6.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 0.5, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0, vl);
  ez = __riscv_vfadd_vf_f64m4(__riscv_vfmul_vv_f64m4(z, ez, vl), 1.0, vl);
  const vuint64m4_t su = __riscv_vsll_vx_u64m4(
      __riscv_vreinterpret_v_i64m4_u64m4(
          __riscv_vadd_vx_i64m4(ki, 1023, vl)), 52, vl);
  const vfloat64m4_t scale = __riscv_vreinterpret_v_u64m4_f64m4(su);
  const vfloat64m4_t res = __riscv_vfmul_vv_f64m4(ez, scale, vl);
  __riscv_vse32_v_f32m2(y + i, __riscv_vfncvt_f_f_w_f32m2(res, vl), vl);
}

// RVV 1.0 fast path for vPowx<float>. See the comment above
// rvv_powx_domain_ok for the accuracy and domain contract.
template <>
inline void vPowx<float>(const int n, const float* a, const float b,
                         float* y) {
  CHECK_GT(n, 0); CHECK(a); CHECK(y);
  if (rvv_powx_domain_ok(a, n, b)) {
    if (b == 1.0f) {
      rvv_map_f32(n, a, y, [](vfloat32m4_t v, size_t vl) { return v; });
      return;
    }
    if (b == 0.0f) {
      rvv_map_f32(n, a, y, [](vfloat32m4_t v, size_t vl) {
        return __riscv_vfmv_v_f_f32m4(1.0f, vl);
      });
      return;
    }
    for (int i = 0; i < n;) {
      const size_t vl = __riscv_vsetvl_e32m2(static_cast<size_t>(n - i));
      rvv_powx_pos_chunk(a, i, n, static_cast<double>(b), y);
      i += static_cast<int>(vl);
    }
    return;
  }
  for (int i = 0; i < n; ++i) { y[i] = pow(a[i], b); }
}
#endif  // defined(__riscv_v) && defined(__riscv_v_intrinsic)

inline void vsPowx(
    const int n, const float* a, const float b, float* y) {
  vPowx<float>(n, a, b, y);
}
inline void vdPowx(
    const int n, const double* a, const float b, double* y) {
  vPowx<double>(n, a, b, y);
}

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
