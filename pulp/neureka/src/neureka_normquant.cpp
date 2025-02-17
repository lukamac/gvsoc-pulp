
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

void Neureka::NormQuantMultSetup(){
  for(int i=0; i<NeurekaTotalPECountXY; i++)
    this->pe_instances[i].SetConfig(reg_config_);
  StreamerConfig streamer_config = this->ctrl_instance.GetNormquantMultStreamerConfig();
  this->normquant_mult_streamer_instance.Init(streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
  this->ctrl_instance.ResetNormQuantMultIteration();
  if(this->trace_config.setup.norm_mult)
    this->trace.msg("Normquant mult Setup is done addr : 0x%x, strides( d0 : 0x%x, d1 : 0x%x, d2 : 0x%x), lengths(d0 : %d, d1 : %d, d2 : %d)\n", streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
  
}

void Neureka::NormQuantShiftSetup(){
  for(int i=0; i<NeurekaTotalPECountXY; i++)
    this->pe_instances[i].SetConfig(reg_config_);
  StreamerConfig streamer_config = this->ctrl_instance.GetNormquantShiftStreamerConfig();
  this->normquant_shift_streamer_instance.Init(streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
  this->ctrl_instance.ResetNormQuantShiftIteration();
  if(this->trace_config.setup.norm_shift)
    this->trace.msg("Normquant shift Setup is done addr : 0x%x, strides( d0 : 0x%x, d1 : 0x%x, d2 : 0x%x), lengths(d0 : %d, d1 : %d, d2 : %d)\n", streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);   
}

void Neureka::NormQuantBiasSetup(){
  StreamerConfig streamer_config = this->ctrl_instance.GetNormquantBiasStreamerConfig();
  this->normquant_bias_streamer_instance.Init(streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);
  this->ctrl_instance.ResetNormQuantBiasIteration();
  if(this->trace_config.setup.norm_bias)
    this->trace.msg("Normquant bias Setup is done addr : 0x%x, strides( d0 : 0x%x, d1 : 0x%x, d2 : 0x%x), lengths(d0 : %d, d1 : %d, d2 : %d)\n", streamer_config.base_addr, streamer_config.stride.d0, streamer_config.stride.d1, streamer_config.stride.d2, streamer_config.length.d0, streamer_config.length.d1, streamer_config.length.d2);    
}

bool Neureka::NormQuantMultExecute(int& latency) {
  // Aliases
  auto &streamer = this->normquant_mult_streamer_instance;
  auto &ctrl = this->ctrl_instance;
  const auto &config0 = this->reg_config_.config0;

  // Fetch raw data
  uint8_t raw_data[L1BandwidthInBytes];
  uint64_t cycles = 0;
  const auto width = ctrl.GetNormQuantMultWidth();
  streamer.VectorLoad(raw_data, width, cycles, this->trace_config.streamer.norm_mult);
  latency += cycles;
  this->num_mem_access_bytes.norm_mult += width; // Just for tracing

  // Transform raw data into scale
  int32_t scale[L1BandwidthInBytes];
  int scale_width;
  std::memset(scale, 0xaa, L1BandwidthInBytes);
  if (config0.normalization_bit_count == 8) {
    scale_width = width;
    for (int i = 0; i < scale_width; i++) {
      scale[i] = (int32_t)(((int8_t *)raw_data)[i]);
    }
  } else if (config0.normalization_bit_count == 32) {
    assert(width % 4 == 0);
    scale_width = width / 4;
    for (int i = 0; i < scale_width; i++) {
      scale[i] = (((int32_t)raw_data[i*4])) +
        (((int32_t)raw_data[i*4+1]) << 8) +
        (((int32_t)raw_data[i*4+2]) << 16) +
        (((int32_t)raw_data[i*4+3]) << 24);
    }
  }

  // Generate shift data
  std::array<OutFeatType, NeurekaAccumulatorPerPECount> shift;
  if (config0.norm_option_bias) {
    // shift in bias step
    std::fill(shift.begin(), shift.end(), 0);
  } else if (config0.norm_option_shift) {
    for (int i = 0; i < scale_width; i++) {
      shift[i] = this->shift_values[i];
    }
  } else {
    std::fill_n(shift.begin(), scale_width, config0.quantization_right_shift);
  }

  // Calculate scaled and optionally shifted result
  // We have offset because of the 32bit mode which processes 8
  // values at time due to bandwidth limitations
  const auto index_offset = ctrl.tiles.index.norm_quant_mult * 8;
  for (auto &pe : this->pe_instances) {
    for (int i = 0; i < scale_width; i++) {
      const auto accum_index = index_offset + i;
      const auto accum = pe.ReadFromIndexAccumBuffer(accum_index);
      // saturating scale
      const int64_t accum_scaled_i64 = (int64_t)accum * (int64_t)scale[i];
      const int64_t accum_scaled_saturated_i48 = std::max((int64_t)(-(1l << 47)), std::min((int64_t)((1l << 47) - 1), accum_scaled_i64));
      const OutFeatType accum_scaled_and_shifted = (int32_t)((accum_scaled_saturated_i48 >> shift[i]) & 0xffffffff);
      pe.WriteAtIndexOnAccumBuffer(accum_index, true, accum_scaled_and_shifted);
    }
  }
  // TODO: Explanation of latency; has to do something with the fact there are 4 multipliers
  // and the bandwidth is 32 bytes to fetch from tcdm...
  latency += 8;

  return ctrl.NormQuantMultIteration();
}

bool Neureka::NormQuantShiftExecute(int& latency){
  int width = this->ctrl_instance.GetNormQuantShiftWidth();
  uint64_t cycles = 0;
  std::array<InFeatType, L1BandwidthInBytes> normquant_shift_8bits = {};
  std::fill(normquant_shift_8bits.begin(), normquant_shift_8bits.end(), 0);
  InFeatType normquant_shift_8bits_temp[width];
  this->normquant_shift_streamer_instance.VectorLoad(normquant_shift_8bits_temp, width, cycles, this->trace_config.streamer.norm_shift);
  for(int i=0; i<width; i++)
    normquant_shift_8bits[i] = normquant_shift_8bits_temp[i]; 
  latency = latency + (int)cycles ? latency + (int)cycles : 1 ;
  this->num_mem_access_bytes.norm_shift += width;


  for(int i=0; i<width; i++){
    int8_t shift = normquant_shift_8bits[i];
    this->shift_values[i] = (int32_t)shift;
  }

  // this->trace.msg("shift accum_0 %d\n", this->pe_instances[0].ReadFromIndexAccumBuffer(0));
  return true;   
}

bool Neureka::NormQuantBiasExecute(int& latency){
  int width = this->ctrl_instance.GetNormQuantBiasWidth();
  uint64_t cycles = 0;
  std::array<InFeatType,32> normquant_bias_8bits={};
  std::array<OutFeatType,32>  normquant_bias_32bits={}; 
  InFeatType normquant_bias_8bits_temp[width];
  this->normquant_bias_streamer_instance.VectorLoad(normquant_bias_8bits_temp, width, cycles, this->trace_config.streamer.norm_bias);
  std::array<OutFeatType, NeurekaAccumulatorPerPECount> accum = this->pe_instances[0].ReadAllAccumBuffer();
  for(int i=0; i<width; i++){
    normquant_bias_8bits[i] = normquant_bias_8bits_temp[i];
    // this->trace.msg("normquant_bias_8bits[%d]:%d\n", i, normquant_bias_8bits[i]);
  }
  latency = latency + (int)cycles ? latency + (int)cycles : 1 ;
  this->num_mem_access_bytes.norm_bias += width;
  
  int index=0;
  for(int i=0; i<width; i=i+4){
    normquant_bias_32bits[index] = (OutFeatType)(normquant_bias_8bits[i+3]<<24) + (normquant_bias_8bits[i+2]<<16) + (normquant_bias_8bits[i+1]<<8) + normquant_bias_8bits[i];
    index++;
  }
  int width_32 = width/4;
  int bias_index = this->ctrl_instance.tiles.index.norm_quant_bias;
  for(int i=0; i<NeurekaTotalPECountXY; i++){
    for(int j=bias_index*8; j<bias_index*8+8; j=j+8){
      std::array<int, 8> index;
      std::array<int, 8> enable;
      std::fill(enable.begin(),enable.end(),1);
      std::array<OutFeatType, 8> norm;
      std::array<OutFeatType, 8> shift;
      for(int iter=0; iter<8; iter++) index[iter] = iter+j;

      for(int k=j; k<8+j; k++){
        norm[k-j] = (k-bias_index*8)<width ? normquant_bias_32bits[k-bias_index*8] : 0;
        if(this->reg_config_.config0.norm_option_shift){
          shift[k-j] = (k-bias_index*8)<width ? this->shift_values[k] : 0;
        }
        else {
          shift[k-j] = (k-bias_index*8)<width ? this->reg_config_.config0.quantization_right_shift : 0;
        }
      }

      this->pe_instances[i].NormQuantBiasShift(index, enable, norm, shift);
    }
  }
  bool done = this->ctrl_instance.NormQuantBiasIteration();
  // this->trace.msg("bias accum_0 %d\n", this->pe_instances[0].ReadFromIndexAccumBuffer(0));
  return done;
    
}
