#ifndef CAFFE_ESPCN_LAYER_HPP_
#define CAFFE_ESPCN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ESPCNLayer : public Layer<Dtype> {
public:
    explicit ESPCNLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "ESPCN"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //                          const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    void shuffle_cpu(Dtype *top, const Dtype *bottom, int top_sp_size, int bottom_sp_size, int bottom_h, int bottom_w);
    void reshuffle_cpu(Dtype *bottom, const Dtype *top, int bottom_sp_size, int top_sp_size, int bottom_h, int bottom_w);

    int upscale_;
    int top_ch_;
    int top_h_;
    int top_w_;
    int group_;
};

}  // namespace caffe

#endif  // CAFFE_ESPCN_LAYER_HPP_
