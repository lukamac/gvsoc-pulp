
/*
 * Copyright (C) 2020-2022  GreenWaves Technologies, ETH Zurich, University of Bologna
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
#include "neureka.hpp"
#include <type_traits>
#include <limits.h>

void Neureka::OutFeatStoreSetup() {
  StreamerConfig streamer_config = this->ctrl_instance.GetOutFeatStoreStreamerConfig2();
  this->outfeat_streamer_instance.Init(streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
  this->ctrl_instance.ResetOutFeatStoreIteration();
  if(this->trace_config.setup.outfeat_store)
    this->trace.msg("OutFeatStore Setup is done addr : 0x%x, strides( d0 : 0x%x, d1 : 0x%x, d2 : 0x%x), lengths(d0 : %d, d1 : %d, d2 : %d)\n", streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
}

void Neureka::ResetAllAccumBuffer(){
  for(int i=0; i<NeurekaTotalPECountXY; i++)
    this->pe_instances[i].ResetAllAccumBuffer();
}

static inline OutFeatType clip(const OutFeatType x, const OutFeatType lo, const OutFeatType hi) {
  return std::max(lo, std::min(hi, x));
}

OutFeatType Neureka::OutFeatQuant(const OutFeatType input){
  if(reg_config_.config0.quantization_bit_count==8 && reg_config_.config0.outfeat_quant)
    if(reg_config_.config0.use_relu || !reg_config_.config0.signed_outfeat){
      return clip(input, 0, 255);
    }else{
      return clip(input, -128, 127);
    }
  else if (reg_config_.config0.quantization_bit_count==32 && reg_config_.config0.outfeat_quant)
    if(reg_config_.config0.use_relu || !reg_config_.config0.signed_outfeat){
      return clip(input, 0, 0xffffffff);
    }else{ 
      return clip(input, INT_MIN, INT_MAX);
    }
  else 
    return input;
}

bool Neureka::OutFeatStoreExecute(int& latency)
{
  int width = this->ctrl_instance.OutFeatStoreWidth();
  int pe_index = this->ctrl_instance.GetOutFeatStoreLinearBufferIndex();// which accumulator buffer to be used
  int word_index = this->ctrl_instance.GetOutFeatStoreWordIndex();
  StreamerDataType store_data[L1BandwidthInBytes];

  // std::cout<<"pe_index="<<pe_index<<"\n";
  if(this->regconfig_manager_instance.reg_config_.config0.quantization_bit_count==32){
    for(int i=0; i<width/4; i++){
        const OutFeatType temp_data = this->pe_instances[pe_index].ReadFromIndexAccumBuffer(word_index+i);
        OutFeatType data = OutFeatQuant(temp_data);
        for(int j=0; j<4; j++){
          store_data[i*4+j] = (data >> (8*j)) & 0xff;
        }
    }
  } else if (this->regconfig_manager_instance.reg_config_.config0.quantization_bit_count==8) {
    for(int i=0; i<width; i++){
        const OutFeatType temp_data = this->pe_instances[pe_index].ReadFromIndexAccumBuffer(i);
        OutFeatType data = OutFeatQuant(temp_data);
        store_data[i] = (StreamerDataType)data; 
    }
  }
  else this->trace.fatal("Unsupported Quantization bit count \n");

  
  uint64_t cycles = 0;

  this->outfeat_streamer_instance.VectorStore(store_data, width, cycles, this->trace_config.streamer.outfeat_store);
  this->num_mem_access_bytes.outfeat_store += width;
  
  this->ctrl_instance.OutFeatStoreIteration2();
  bool streamout_done = this->ctrl_instance.load_store_status.outfeat.done;
  if(streamout_done){
    ResetAllAccumBuffer();
  }

  return streamout_done;
}
