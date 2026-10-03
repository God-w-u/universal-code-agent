#include "ring_buffer.h"

// 构造：按容量开好底层数组，游标和计数都从零开始。
RingBuffer::RingBuffer(std::size_t capacity) : data_(capacity) {}

void RingBuffer::push(double value) {
    data_[head_] = value;
    head_ = (head_ + 1) % data_.size();
    if (count_ < data_.size()) {
        ++count_;
    }
    ++total_;
}

bool RingBuffer::pop(double& out) {
    if (count_ == 0) {
        return false;
    }
    // 最旧的一条在 head_ 往前数 count_ 个的位置。
    const std::size_t tail = (head_ + data_.size() - count_) % data_.size();
    out = data_[tail];
    --count_;
    return true;
}

std::size_t RingBuffer::size() const { return count_; }

std::size_t RingBuffer::total_written() const { return total_; }

void RingBuffer::clear() {
    head_ = 0;
    count_ = 0;
    total_ = 0;
}
