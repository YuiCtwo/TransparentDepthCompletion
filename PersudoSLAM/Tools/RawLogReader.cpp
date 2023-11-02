/*
* This file is part of ElasticFusion.
*
* Copyright (C) 2015 Imperial College London
*
* The use of the code within this file and all code within files that
* make up the software that is ElasticFusion is permitted for
* non-commercial purposes only.  The full terms and conditions that
* apply to the code within this file are detailed within the LICENSE.txt
* file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
* unless explicitly stated.  By downloading this file you agree to
* comply with these terms.
*
* If you wish to use any of this code for commercial purposes then
* please email researchcontracts.engineering@imperial.ac.uk.
*
*/

#include "RawLogReader.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <chrono>
#include <string>
#include <ctime>

void eval_time_used(clock_t start) {

  clock_t t = clock();
  std::cout << "Time used: " << (double)(t - start) / CLOCKS_PER_SEC << std::endl;
}


RawLogReader::RawLogReader(std::string file, bool flipColors)
    : RawLogReader(file, flipColors, 787.324, 787.324, 320.0, 240.0) {
}

RawLogReader::RawLogReader(std::string file, bool flipColors, float fx, float fy, float cx, float cy)
   : LogReader(file, flipColors)
{
 assert(pangolin::FileExists(file.c_str()));

 fp = fopen(file.c_str(), "rb");
 currentFrame = 0;

 auto tmp = fread(&numFrames,sizeof(int32_t),1,fp);
 assert(tmp);

 depthReadBuffer = new unsigned char[numPixels * 2];
 imageReadBuffer = new unsigned char[numPixels * 3];
 nextImageReadBuffer = new unsigned char[numPixels * 3];
 maskReadBuffer = new unsigned char[numPixels * 3];
 HBuffer = new unsigned char[4*4*4];  // float: 32bit
 decompressionBufferNextImage = new Bytef[Resolution::getInstance().numPixels() * 3];
 decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];
 decompressionBufferImage = new Bytef[Resolution::getInstance().numPixels() * 3];
 decompressionBufferMask = new Bytef[Resolution::getInstance().numPixels() * 3];
 decompressionBufferH = new Bytef[16*4];

 // TODO: fill camrea param
 this->fx = torch::tensor({fx}).to(torch::kCUDA);
 this->fy = torch::tensor({fy}).to(torch::kCUDA);
 this->cx = torch::tensor({cx}).to(torch::kCUDA);
 this->cy = torch::tensor({cy}).to(torch::kCUDA);

 // load_modle();
 // pytorch_module = torch::jit::load("/home/mathloverpi/code/cpp_SINET/cpp/model_cpp.pt");
 // std::ifstream myfile ("/home/mathloverpi/code/ElasticFusion/cpp/model_cpp.pt", std::ifstream::in);
 pytorch_modle = torch::jit::load("/home/ctwo/VDCNet/pretrained/df_L50_cpp_gpu.pth", torch::kCUDA);
// pytorch_modle.to(torch::kCUDA);
 pytorch_modle.eval();
}

RawLogReader::~RawLogReader()
{
 delete [] depthReadBuffer;
 delete [] imageReadBuffer;
 delete [] maskReadBuffer;
 delete [] nextImageReadBuffer;
 delete [] decompressionBufferDepth;
 delete [] decompressionBufferImage;
 delete [] decompressionBufferMask;
 delete [] decompressionBufferNextImage;
 delete [] HBuffer;
 delete [] decompressionBufferH;
 fclose(fp);
}

void RawLogReader::getBack()
{
 assert(filePointers.size() > 0);

 fseek(fp, filePointers.top(), SEEK_SET);

 filePointers.pop();

 getCore();
}

void RawLogReader::getNext()
{
 filePointers.push(ftell(fp));

 getCore();
}

void RawLogReader::getCore()
{
 auto tmp = fread(&timestamp,sizeof(int64_t),1,fp);
 assert(tmp);
 tmp = fread(&depthSize,sizeof(int32_t),1,fp);
 assert(tmp);
 tmp = fread(&imageSize,sizeof(int32_t),1,fp);
 assert(tmp);
 tmp = fread(&nextImgSize,sizeof(int32_t),1,fp);
 assert(tmp);
 tmp = fread(&maskSize,sizeof(int32_t),1,fp);
 assert(tmp);
 tmp = fread(&HSize, sizeof(int32_t),1,fp);
 assert(tmp);
 tmp = fread(depthReadBuffer,depthSize,1,fp);
 assert(tmp);
 if(imageSize > 0) {
   tmp = fread(imageReadBuffer,imageSize,1,fp);
   assert(tmp);
 }
 if (nextImgSize > 0) {
   tmp = fread(nextImageReadBuffer, nextImgSize, 1, fp);
   assert(tmp);
 }
 if(maskSize > 0)
 {
   tmp = fread(maskReadBuffer, maskSize, 1, fp);
   assert(tmp);
 }
 tmp = fread(HBuffer,HSize,1,fp);

 if(depthSize == numPixels * 2)
 {
   memcpy(&decompressionBufferDepth[0], depthReadBuffer, numPixels * 2);
 }
 else
 {
   unsigned long decompLength = numPixels * 2;
   uncompress(&decompressionBufferDepth[0], (unsigned long *)&decompLength, (const Bytef *)depthReadBuffer, depthSize);
 }

 // read cur and next image
 if(imageSize == numPixels * 3)
 {
   std::cout << 1 <<std::endl;
   memcpy(&decompressionBufferImage[0], imageReadBuffer, numPixels * 3);
   memcpy(&decompressionBufferNextImage[0], nextImageReadBuffer, numPixels * 3);

 }
 else if((imageSize > 0) && (nextImgSize > 0))
 {
   jpeg.readData(imageReadBuffer, imageSize, (uint8_t *)&decompressionBufferImage[0]);
   jpeg.readData(nextImageReadBuffer, nextImgSize, (uint8_t *)&decompressionBufferImage[0]);

 }
 else
 {
   memset(&decompressionBufferImage[0], 0, numPixels * 3);
   memset(&decompressionBufferNextImage[0], 0, numPixels * 3);
 }

 // read mask
 if(maskSize == numPixels * 3)
 {
   memcpy(&decompressionBufferMask[0], maskReadBuffer, numPixels);

 }
 else if(maskSize > 0)
 {
   jpeg.readData(maskReadBuffer, maskSize, (uint8_t *)&decompressionBufferMask[0]);
 }
 else
 {
   memset(&decompressionBufferMask[0], 0, numPixels);
 }
 // read H mat
 memcpy(&decompressionBufferH[0], HBuffer, 16*4);
 depth = (unsigned short *)decompressionBufferDepth;
 rgb = (unsigned char *)&decompressionBufferImage[0];
 next_rgb = (unsigned char *)&decompressionBufferNextImage[0];
 mask = (unsigned char *)&decompressionBufferMask[0];
 H = (float *)decompressionBufferH;

 cv::Mat cv_mask(480,640, CV_8UC3, (cv::Vec<unsigned char, 3> *)mask);
 cv::Mat cv_mask_resized(224,224, CV_8UC1);

 cv::Mat cv_cur_rgb(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)rgb);
 cv::Mat cv_next_rgb(480,640,CV_8UC3,(cv::Vec<unsigned char, 3> *)next_rgb);
 cv::Mat cv_cur_rgb_resized(224,224,CV_8UC3);
 cv::Mat cv_next_rgb_resized(224,224,CV_8UC3);
 cv::Mat cv_depth(480,640, CV_16UC1,(unsigned short *)&depth[0]);
 cv::Mat cv_depth_resized(224,224, CV_16UC1);
 cv::Mat cv_H(4, 4, CV_32F, &decompressionBufferH[0]);
 cv_depth = cv_depth / 1000;


 cv::cvtColor(cv_mask, cv_mask, cv::COLOR_BGR2GRAY);
 cv_mask.convertTo(cv_mask, CV_8UC1);
// std::cout << cv::format(cv_mask, cv::Formatter::FMT_NUMPY) << std::endl;
 // resize
 cv::Size scale(224, 224);
 cv::resize(cv_mask, cv_mask_resized, scale, 0, 0, cv::INTER_LINEAR);
 cv::resize(cv_cur_rgb, cv_cur_rgb_resized, scale, 0, 0, cv::INTER_LINEAR);
 cv::resize(cv_next_rgb, cv_next_rgb_resized, scale, 0, 0, cv::INTER_LINEAR);
 cv::resize(cv_depth, cv_depth_resized, scale, 0, 0, cv::INTER_LINEAR);

 threshold(cv_mask_resized, cv_mask_resized, 100, 1, cv::THRESH_BINARY);
 cv_depth_resized.convertTo(cv_depth_resized, CV_32F);
 double depth_max_val, depth_min_val;
 cv::minMaxIdx(cv_depth_resized, &depth_min_val, &depth_max_val,
               nullptr, nullptr, 1-cv_mask_resized);
 cv_depth_resized = cv_depth_resized / depth_max_val;
//std::cout << depth_max_val << std::endl;
//std::cout << depth_min_val << std::endl;
// std::cout << cv::format(cv_depth, cv::Formatter::FMT_NUMPY) << std::endl;
// std::cout << cv::format(cv_depth_resized, cv::Formatter::FMT_NUMPY) << std::endl;
 // opencv to tensor()
 // R, t
 torch::Tensor tensor_H = torch::from_blob(cv_H.data, {4, 4});
 torch::Tensor R_mat = tensor_H.index({
     torch::indexing::Slice({torch::indexing::None, 3}),
     torch::indexing::Slice({torch::indexing::None, 3})}
 ); // H[:3, :3]
 torch::Tensor t_vec = tensor_H.index(
     {torch::indexing::Slice({torch::indexing::None, 3}),
      3}
 ); // H[:3, 3]
 R_mat = torch::unsqueeze(R_mat, 0);
 t_vec = torch::unsqueeze(t_vec, 0);
// std::cout << R_mat << std::endl;
// std::cout << t_vec << std::endl;
 // rgb
 cv_cur_rgb_resized.convertTo(cv_cur_rgb_resized, CV_32FC3, 1.0 / 255.0);
 cv_next_rgb_resized.convertTo(cv_next_rgb_resized, CV_32FC3, 1.0 / 255.0);
 torch::Tensor tensor_cur_rgb = torch::from_blob(cv_cur_rgb_resized.data, {1, cv_cur_rgb_resized.rows, cv_cur_rgb_resized.cols,3});
 torch::Tensor tensor_next_rgb = torch::from_blob(cv_next_rgb_resized.data, {1, cv_next_rgb_resized.rows, cv_cur_rgb_resized.cols,3});
 tensor_cur_rgb = tensor_cur_rgb.permute({0, 3, 1, 2});
 tensor_next_rgb = tensor_next_rgb.permute({0, 3, 1, 2});
 // transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 tensor_cur_rgb[0][0] = tensor_cur_rgb[0][0].sub_(0.485).div_(0.229);
 tensor_cur_rgb[0][1] = tensor_cur_rgb[0][1].sub_(0.456).div_(0.224);
 tensor_cur_rgb[0][2] = tensor_cur_rgb[0][2].sub_(0.406).div_(0.225);
 tensor_next_rgb[0][0] = tensor_next_rgb[0][0].sub_(0.485).div_(0.229);
 tensor_next_rgb[0][1] = tensor_next_rgb[0][1].sub_(0.456).div_(0.224);
 tensor_next_rgb[0][2] = tensor_next_rgb[0][2].sub_(0.406).div_(0.225);
 // mask
 torch::Tensor tensor_mask = torch::from_blob(cv_mask_resized.data,
                                              {1, 1, cv_mask_resized.rows, cv_mask_resized.cols});
 // depth
 torch::Tensor tensor_depth = torch::from_blob(cv_depth_resized.data,
                                               {1, 1, cv_depth_resized.rows, cv_depth_resized.cols});
 tensor_depth = tensor_depth / 4000;
// tensor_depth = torch::where(tensor_mask > 0, 0, tensor_depth);
 // fx, fy, cx, cy already store in class
 // scale
 torch::Tensor depth_scale = torch::ones({1}).fill_(depth_max_val).to(torch::kFloat);
// std::cout << depth_scale << std::endl;

 // min_depth
 torch::Tensor neg_mask = torch::where(tensor_mask > 0, 0, 1).to(torch::kBool);
// torch::Tensor min_depth = torch::zeros({1, 1, 240, 320}).fill_(depth_min_val);
// std::cout << depth_min_val << std::endl;

 // to device
 tensor_cur_rgb = tensor_cur_rgb.to(torch::kCUDA);
 tensor_next_rgb = tensor_next_rgb.to(torch::kCUDA);
 tensor_mask = tensor_mask.to(torch::kCUDA);
 tensor_depth = tensor_depth.to(torch::kCUDA);
// min_depth = min_depth.to(torch::kCPU);
 R_mat = R_mat.to(torch::kCUDA);
 t_vec = t_vec.to(torch::kCUDA);
 depth_scale = depth_scale.to(torch::kCUDA);
// clock_t start = clock();
 torch::NoGradGuard no_grad;
 // tensor_mask, tensor_depth
 auto result = pytorch_modle.forward({
     tensor_cur_rgb, tensor_next_rgb, R_mat, t_vec, depth_scale,
     fx, fy, cx, cy}).toTensor();
//
// result = result.to(torch::kCPU).to(torch::kByte);
// eval_time_used(start);
// cv::Mat predict_depth(scale, CV_16UC1, result.data_ptr());
// predict_depth = predict_depth * depth_factor;
// std::cout << cv::format(cv_depth, cv::Formatter::FMT_PYTHON) << std::endl;
// resize(predict_depth, predict_depth, cv::Size(640, 480));

 // mask, set 1=transparent object
 threshold(cv_mask, cv_mask, 128, 1, cv::THRESH_BINARY);
 // copy non-zero value
// predict_depth.copyTo(cv_depth, cv_mask);
 memcpy(&decompressionBufferDepth[0], (char*)cv_depth.data, numPixels * 2);

 if(flipColors)
 {
   for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
   {
     std::swap(rgb[i+0], rgb[i+2]);
   }
 }
 currentFrame++;
}

void RawLogReader::fastForward(int frame)
{
 while(currentFrame < frame && hasMore())
 {
   filePointers.push(ftell(fp));

   auto tmp = fread(&timestamp,sizeof(int64_t),1,fp);
   assert(tmp);
   tmp = fread(&depthSize,sizeof(int32_t),1,fp);
   assert(tmp);
   tmp = fread(&imageSize,sizeof(int32_t),1,fp);
   assert(tmp);
   tmp = fread(&nextImgSize,sizeof(int32_t),1,fp);
   assert(tmp);
   tmp = fread(&maskSize,sizeof(int32_t),1,fp);
   assert(tmp);
   tmp = fread(&HSize, sizeof(int32_t),1,fp);
   assert(tmp);
   tmp = fread(depthReadBuffer,depthSize,1,fp);
   assert(tmp);
   if(imageSize > 0)
   {
     tmp = fread(imageReadBuffer,imageSize,1,fp);
     assert(tmp);
     tmp = fread(nextImageReadBuffer, nextImgSize, 1, fp);
     assert(tmp);
   }
   if(maskSize > 0)
   {
     tmp = fread(maskReadBuffer, maskSize, 1, fp);
     assert(tmp);
   }
   tmp = fread(HBuffer,HSize,1,fp);
   assert(tmp);
   currentFrame++;
 }
}

int RawLogReader::getNumFrames()
{
 return numFrames;
}

bool RawLogReader::hasMore()
{
 return currentFrame + 1 < numFrames;
}


void RawLogReader::rewind()
{
 if (filePointers.size() != 0)
 {
   std::stack<int> empty;
   std::swap(empty, filePointers);
 }

 fclose(fp);
 fp = fopen(file.c_str(), "rb");

 auto tmp = fread(&numFrames,sizeof(int32_t),1,fp);
 assert(tmp);

 currentFrame = 0;
}

bool RawLogReader::rewound()
{
 return filePointers.size() == 0;
}

const std::string RawLogReader::getFile()
{
 return file;
}

void RawLogReader::setAuto(bool value)
{

}
