#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

namespace caffe {

using std::min;
using std::max;

// Scalar MAX-pooling reduction over output columns [pw_begin, pw_end) of a
// single output row `ph`. This is the original per-pixel loop, factored out
// so that both the fully scalar path and the RVV interior/border split share
// exactly the same tie / NaN / index semantics.
template <typename Dtype>
static inline void max_pool_scalar_range(const Dtype* bottom_data,
    const int width, const int ph, const int pw_begin, const int pw_end,
    const int hstart, const int hend, const int pad_w, const int kernel_w,
    const int stride_w, const int pooled_width, Dtype* top_data,
    Dtype* top_mask, int* mask) {
  for (int pw = pw_begin; pw < pw_end; ++pw) {
    int wstart = pw * stride_w - pad_w;
    const int wend = std::min(wstart + kernel_w, width);
    wstart = std::max(wstart, 0);
    const int pool_index = ph * pooled_width + pw;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int index = h * width + w;
        if (bottom_data[index] > top_data[pool_index]) {
          top_data[pool_index] = bottom_data[index];
          if (top_mask != NULL) {
            top_mask[pool_index] = static_cast<Dtype>(index);
          } else {
            mask[pool_index] = index;
          }
        }
      }
    }
  }
}

// Generic (scalar) MAX-pooling of one (n, c) plane. Used for every Dtype that
// has no explicit vectorized overload below.
template <typename Dtype>
static void max_pool_forward_plane(const Dtype* bottom_data, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, Dtype* top_data, Dtype* top_mask,
    int* mask) {
  for (int ph = 0; ph < pooled_height; ++ph) {
    const int hstart_raw = ph * stride_h - pad_h;
    const int hstart = std::max(hstart_raw, 0);
    const int hend = std::min(hstart_raw + kernel_h, height);
    max_pool_scalar_range<Dtype>(bottom_data, width, ph, 0, pooled_width,
        hstart, hend, pad_w, kernel_w, stride_w, pooled_width, top_data,
        top_mask, mask);
  }
}

#if defined(__riscv_v)
// RVV 1.0 MAX-pooling of one (n, c) float plane.
//
// For an output row whose kernel window is fully inside the image height and
// for output columns whose window is fully inside the image width, the window
// taps shift by exactly `stride_w` per output column. Those interior columns
// are processed as one vector across pw: vlse32.v gathers each tap with a
// byte stride of stride_w * 4, vmfgt.vv produces the strict-greater mask and
// vmerge.vvm updates both the running maximum and the running index. Ties are
// therefore resolved to the first (row-major) occurrence exactly like the
// scalar loop, NaN taps never update (comparison is false) and lanes that
// never update keep -FLT_MAX / -1. Border and tail columns keep the scalar
// path. SEW=e32, LMUL=m4, dynamic vsetvl.
static __attribute__((noinline)) void max_pool_forward_plane(
    const float* bottom_data, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, float* top_data, float* top_mask,
    int* mask) {
  // The horizontal interior is identical for every output row.  Computing it
  // once avoids repeated integer divisions for every row of every plane.
  int interior_pwl = 0;
  if (pad_w > 0) {
    interior_pwl = std::min((pad_w + stride_w - 1) / stride_w, pooled_width);
  }
  const int limit = width + pad_w - kernel_w;
  int interior_pwr = limit < 0 ? 0 : limit / stride_w + 1;
  interior_pwr = std::min(interior_pwr, pooled_width);
  interior_pwl = std::min(interior_pwl, interior_pwr);
  const ptrdiff_t byte_stride =
      static_cast<ptrdiff_t>(stride_w) * sizeof(float);

  for (int ph = 0; ph < pooled_height; ++ph) {
    const int hstart_raw = ph * stride_h - pad_h;
    const int hstart = std::max(hstart_raw, 0);
    const int hend = std::min(hstart_raw + kernel_h, height);
    const bool row_interior =
        (hstart_raw >= 0) && (hstart_raw + kernel_h <= height);

    // Interior output columns [pwl, pwr): window fully inside [0, width).
    const int pwl = row_interior ? interior_pwl : 0;
    const int pwr = row_interior ? interior_pwr : 0;

    max_pool_scalar_range<float>(bottom_data, width, ph, 0, pwl, hstart, hend,
        pad_w, kernel_w, stride_w, pooled_width, top_data, top_mask, mask);

    const int n = pwr - pwl;
    if (n > 0) {
      for (int i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(static_cast<size_t>(n - i));
        const int pw0 = pwl + i;
        const int wstart0 = pw0 * stride_w - pad_w;
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vl);
        vint32m4_t vindex = __riscv_vmv_v_x_i32m4(-1, vl);
        const vuint32m4_t lane = __riscv_vid_v_u32m4(vl);
        const vint32m4_t lane_off = __riscv_vreinterpret_v_u32m4_i32m4(
            __riscv_vmul_vx_u32m4(lane, static_cast<uint32_t>(stride_w), vl));
        const auto accumulate_tap = [&](const int index_base,
                                        const int tap_offset) {
          const vfloat32m4_t tap = __riscv_vlse32_v_f32m4(
              bottom_data + index_base, byte_stride, vl);
          const vbool8_t gt = __riscv_vmfgt_vv_f32m4_b8(tap, vmax, vl);
          vmax = __riscv_vmerge_vvm_f32m4(vmax, tap, gt, vl);
          vindex = __riscv_vmerge_vxm_i32m4(
              vindex, tap_offset, gt, vl);
        };
        if (kernel_w == 3) {
          for (int h = hstart; h < hstart + kernel_h; ++h) {
            const int tap_offset = (h - hstart) * width;
            const int row = h * width + wstart0;
            accumulate_tap(row, tap_offset);
            accumulate_tap(row + 1, tap_offset + 1);
            accumulate_tap(row + 2, tap_offset + 2);
          }
        } else {
          for (int h = hstart; h < hstart + kernel_h; ++h) {
            for (int kw = 0; kw < kernel_w; ++kw) {
              accumulate_tap(h * width + wstart0 + kw,
                  (h - hstart) * width + kw);
            }
          }
        }
        const vbool8_t valid = __riscv_vmsge_vx_i32m4_b8(vindex, 0, vl);
        vint32m4_t absolute_index =
            __riscv_vadd_vv_i32m4(vindex, lane_off, vl);
        absolute_index = __riscv_vadd_vx_i32m4(
            absolute_index, hstart * width + wstart0, vl);
        vindex = __riscv_vmerge_vvm_i32m4(vindex, absolute_index, valid, vl);
        const int pool0 = ph * pooled_width + pw0;
        __riscv_vse32_v_f32m4(top_data + pool0, vmax, vl);
        if (top_mask != NULL) {
          __riscv_vse32_v_f32m4(top_mask + pool0,
              __riscv_vfcvt_f_x_v_f32m4(vindex, vl), vl);
        } else {
          __riscv_vse32_v_i32m4(mask + pool0, vindex, vl);
        }
        i += static_cast<int>(vl);
      }
    }

    max_pool_scalar_range<float>(bottom_data, width, ph, pwr, pooled_width,
        hstart, hend, pad_w, kernel_w, stride_w, pooled_width, top_data,
        top_mask, mask);
  }
}
#endif  // defined(__riscv_v)

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        max_pool_forward_plane(bottom_data, height_, width_, pooled_height_,
            pooled_width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_,
            stride_w_, top_data, top_mask, mask);
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
