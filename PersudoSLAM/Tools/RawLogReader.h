/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at
 * <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef RAWLOGREADER_H_
#define RAWLOGREADER_H_

#include <pangolin/utils/file_utils.h>
#include "../Core/Utils/Resolution.h"
#include "../Core/Utils/Stopwatch.h"

#include "LogReader.h"

//#include <opencv2/core/mat.hpp>
#include <stdio.h>
#include <zlib.h>
#include <cassert>
#include <iostream>
#include <stack>
#include <string>
#include "torch/script.h"
#include "torch/torch.h"

class RawLogReader : public LogReader {
 public:
  RawLogReader(std::string file, bool flipColors);

  RawLogReader(std::string file, bool flipColors, float fx, float fy, float cx, float cy);

  virtual ~RawLogReader();

  void getNext();

  void getBack();

  int getNumFrames();

  bool hasMore();

  bool rewound();

  void rewind();

  void fastForward(int frame);

  const std::string getFile();

  void setAuto(bool value);

  std::stack<int> filePointers;
  int depth_factor = 500;

 private:
  void getCore();
  torch::jit::Module pytorch_modle;
  torch::Tensor fx, fy, cx, cy;
//  torch::Tensor tensor_cur_rgb;
//  torch::Tensor tensor_next_rgb;
//  torch::Tensor tensor_mask;
//  torch::Tensor tensor_depth;
//  torch::Tensor min_depth;
//  torch::Tensor depth_scale;
//  torch::Tensor R_mat;
//  torch::Tensor t_vec;
//  torch::Tensor tensor_H;

 protected:
  Bytef* decompressionBufferNextImage;
  Bytef* decompressionBufferMask;
  Bytef* decompressionBufferH;
  uint8_t* nextImageReadBuffer;
  uint8_t* maskReadBuffer;
  uint8_t* HBuffer;
  int32_t maskSize;
  int32_t nextImgSize;
  int32_t HSize;
  uint8_t* next_rgb;
  uint8_t* mask;
  float *H;
};

#endif /* RAWLOGREADER_H_ */
