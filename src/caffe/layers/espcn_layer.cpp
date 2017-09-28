#include <algorithm>
#include <vector>

#include "caffe/layers/espcn_layer.hpp"

namespace caffe {

template <typename Dtype>
void ESPCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    upscale_ = this->layer_param_.espcn_param().upscale();
    CHECK_GT(upscale_, 0) << "upscale must be greater than 0";

    const int chs = bottom[0]->channels();
    group_ = upscale_ * upscale_;
    top_ch_ = chs / group_;
    top_h_ = bottom[0]->height() * upscale_;
    top_w_ = bottom[0]->width() * upscale_;
    CHECK_EQ(chs, top_ch_ * group_)
            << "bottom channel number should be divided with no remainder by upscale_ * upscale_";

    top[0]->Reshape(bottom[0]->num(), top_ch_, top_h_, top_w_);
}

template <typename Dtype>
void ESPCNLayer<Dtype>::shuffle_cpu(Dtype *top, const Dtype *bottom,
                                    int top_sp_size, int bottom_sp_size,
                                    int bottom_h, int bottom_w)
{
    for (int g = 0; g < group_; ++g) {
        for (int ch = 0; ch < top_ch_; ++ch) {
            for (int r = 0; r < bottom_h; ++r) {
                for (int c = 0; c < bottom_w; ++c) {
                    int top_r = r * upscale_ + g / upscale_;
                    int top_c = c * upscale_ + g % upscale_;
                    top[ch * top_sp_size + top_r * top_w_ + top_c] =
                    bottom[(ch * group_ + g) * bottom_sp_size + r * bottom_w + c];
                }
            }
        }
    }
}

template <typename Dtype>
void ESPCNLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  top_h_ = bottom[0]->height() * upscale_;
  top_w_ = bottom[0]->width() * upscale_;

  top[0]->Reshape(bottom[0]->num(), top_ch_, top_h_, top_w_);

}

template <typename Dtype>
void ESPCNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const int num = bottom[0]->shape(0);

    for(int n = 0; n < num; ++n) {
        shuffle_cpu(top_data + n * top[0]->count(1),
                    bottom_data + n * bottom[0]->count(1),
                    top[0]->count(2), bottom[0]->count(2),
                    bottom[0]->height(),
                    bottom[0]->width());
    }
}

template <typename Dtype>
void ESPCNLayer<Dtype>::reshuffle_cpu(Dtype *bottom, const Dtype *top,
                                      int bottom_sp_size, int top_sp_size,
                                      int bottom_h, int bottom_w) {
    for (int g = 0; g < group_; ++g) {
        for (int ch = 0; ch < top_ch_; ++ch) {
            for (int r = 0; r < bottom_h; ++r) {
                for (int c = 0; c < bottom_w; ++c) {
                    int top_r = r * upscale_ + g / upscale_;
                    int top_c = c * upscale_ + g % upscale_;
                    bottom[(ch * group_ + g) * bottom_sp_size + r * bottom_w + c] =
                    top[ch * top_sp_size + top_r * top_w_ + top_c];
                }
            }
        }
    }
}

template <typename Dtype>
void ESPCNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        const int num = bottom[0]->shape(0);

        for(int n = 0; n < num; ++n) {
            reshuffle_cpu(bottom_diff + n * bottom[0]->count(1),
                          top_diff + n * top[0]->count(1),
                          bottom[0]->count(2), top[0]->count(2),
                          bottom[0]->height(), bottom[0]->width());
        }
    }
}


#ifdef CPU_ONLY
STUB_GPU(ESPCNLayer);
#endif

INSTANTIATE_CLASS(ESPCNLayer);
REGISTER_LAYER_CLASS(ESPCN);
}  // namespace caffe
