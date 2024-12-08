
/*
 * Copyright (C) 2020-2022  GreenWaves Technologies, ETH Zurich, University of
 * Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Authors: Arpan Suravi Prasad, ETH Zurich (prasadar@iis.ee.ethz.ch)
 */

#ifndef __STREAMER_H__
#define __STREAMER_H__

#include "datatype.hpp"
#include "vp/itf/io.hpp"
#include <algorithm>

static const AddrType alignment = 4;

template <int BandWidth> class Streamer {

public:
  void VectorStore(uint8_t* data, int size, uint64_t& cycles, bool verbose);
  void VectorLoad(uint8_t* data, int size, uint64_t& cycles, bool verbose);

  Streamer() {};

  Streamer(vp::IoMaster* io_master, vp::IoReq* io_req, vp::Trace* trace) {
    io_req = io_req;
    io_master = io_master;
    trace = trace;
  }

  void Init(AddrType baseAddr, int d0Stride, int d1Stride, int d2Stride,
            int d0Length, int d1Length, int d2Length) {
    base_addr_ = baseAddr;
    d0_stride_ = d0Stride;
    d1_stride_ = d1Stride;
    d2_stride_ = d2Stride;
    d0_length_ = d0Length;
    d1_length_ = d1Length;
    d2_length_ = d2Length;
    ResetCount();
  }

private:
  vp::IoMaster* io_master;
  vp::IoReq* io_req;
  vp::Trace* trace;

  AddrType base_addr_;
  int d0_stride_, d1_stride_, d2_stride_;
  int d0_length_, d1_length_, d2_length_;
  int d0_count_, d1_count_, d2_count_;

  void UpdateCount();
  void ResetCount();
  AddrType ComputeAddressOffset() const;
  AddrType ComputeAddress() const;
  inline void SingleBankTransaction(AddrType address, uint8_t* data,
                                    bool is_write, uint64_t& cycles,
                                    uint64_t& max_latency, bool verbose);
  inline void VectorTransaction(uint8_t* data, int size, bool is_write,
                                uint64_t& cycles, bool verbose);
};

template <int BandWidth> void Streamer<BandWidth>::ResetCount() {
  d0_count_ = 0;
  d1_count_ = 0;
  d2_count_ = 0;
}

template <int BandWidth> void Streamer<BandWidth>::UpdateCount() {
  d0_count_++;
  if (d0_count_ == d0_length_) {
    d0_count_ = 0;
    d1_count_++;
    if (d1_count_ == d1_length_) {
      d1_count_ = 0;
      d2_count_++;
      if (d2_count_ == d2_length_) {
        d2_count_ = 0;
      }
    }
  }
}

template <int BandWidth>
AddrType Streamer<BandWidth>::ComputeAddressOffset() const {
  return d2_count_ * d2_stride_ + d1_count_ * d1_stride_ +
         d0_count_ * d0_stride_;
}

template <int BandWidth> AddrType Streamer<BandWidth>::ComputeAddress() const {
  return base_addr_ + ComputeAddressOffset();
}

template <int BandWidth>
void inline Streamer<BandWidth>::SingleBankTransaction(
    AddrType address, uint8_t* data, bool is_write, uint64_t& cycles,
    uint64_t& max_latency, bool verbose) {
  assert(address % alignment == 0 &&
         "Only aligned addresses are allowed in SingleBankTransaction");
  const int size = 4;
  *io_req = vp::IoReq(address, data, size, is_write);
  const auto status = io_master->req(io_req);

  if (status != vp::IO_REQ_OK) {
    trace->fatal("Unsupported asynchronous reply\n");
  }

  max_latency = std::max(max_latency, io_req->get_latency());

  if (verbose) {
    trace->msg(
        "max_latency = %llu, Address =%08x, size=%x, latency=%llu, we=%d, "
        "data[0]=%02x, data[1]=%02x, data[2]=%02x, data[3]=%02x\n",
        max_latency, address, size, io_req->get_latency(), is_write, data[0],
        data[1], data[2], data[3]);
  }
}

// Only for single load transaction. So the size should be less than the
// bandwidth
template <int BandWidth>
void Streamer<BandWidth>::VectorTransaction(uint8_t* data, int size,
                                            bool is_write, uint64_t& cycles,
                                            bool verbose) {
  uint64_t max_latency = 0;
  const AddrType addr = ComputeAddress();
  const AddrType addr_start_offset = addr % alignment;
  const AddrType addr_start_aligned = addr - addr_start_offset;

  // Increase the size by the start offset so that the start is aligned
  int size_aligned = size + addr_start_offset;

  // Increase the size by what's missing to have an aligned load at the end too
  const int addr_end_offset = size_aligned % alignment;
  if (addr_end_offset > 0) {
    size_aligned += alignment - addr_end_offset;
  }

  assert(size_aligned % alignment == 0 && "size_aligned is not aligned");
  assert(size_aligned <= BandWidth + 8 &&
         "size is larger than the BandWidth + 8");

  // Do the aligned fetch
  uint8_t data_aligned[BandWidth];
  uint8_t* data_ptr = data_aligned;
  AddrType addr_aligned = addr_start_aligned;
  for (int i = 0; i < size_aligned / alignment; i++) {
    SingleBankTransaction(addr_aligned, data_ptr, is_write, cycles, max_latency,
                          verbose);
    data_ptr += alignment;
    addr_aligned += alignment;
  }

  // Extract only the data we need
  memcpy(data, &data_aligned[addr_start_offset], size);

  cycles += max_latency + 1;
  if (verbose) {
    trace->msg(" cycles : %llu, max_latency : %d\n", cycles, max_latency);
  }

  UpdateCount();
}

template <int BandWidth>
void Streamer<BandWidth>::VectorLoad(uint8_t* data, int size, uint64_t& cycles,
                                     bool verbose) {
  const bool is_write = true;
  VectorTransaction(data, size, is_write, cycles, verbose);
}

template <int BandWidth>
void Streamer<BandWidth>::VectorStore(uint8_t* data, int size, uint64_t& cycles,
                                      bool verbose) {
  const bool is_write = false;
  VectorTransaction(data, size, is_write, cycles, verbose);
}

#endif // __STREAMER_H__
