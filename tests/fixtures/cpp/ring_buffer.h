// 环形缓冲区：容量固定，写满之后新数据覆盖最旧的一条。
// 用法：push 写入、pop 按写入顺序取出。
#pragma once

#include <cstddef>
#include <vector>

class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity);

    // 写入一条数据。缓冲已经满了的话，覆盖最旧的那条。
    void push(double value);

    // 按写入顺序取出最旧的一条；缓冲为空时返回 false。
    bool pop(double& out);

    // 当前缓冲里存了多少条数据。
    std::size_t size() const;

    // 从创建到现在一共写入过多少条（含已经被覆盖掉的）。
    std::size_t total_written() const;

    // 清空缓冲，容量不变。
    void clear();

private:
    std::vector<double> data_;
    std::size_t head_ = 0;   // 下一条要写入的位置
    std::size_t count_ = 0;  // 当前条数
    std::size_t total_ = 0;  // 累计写入条数
};
