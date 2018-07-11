
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define THREAD_BLOCK_DIM 16


__global__ void rgbaToGrey(
	const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	int rows,
	int cols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((x < cols) && (y < rows)) {
		int position = y * cols + x;
		greyImage[position] = rgbaImage[position].x * 0.299f + rgbaImage[position].y * 0.587f + rgbaImage[position].z * 0.114f;
	}
}

void runRGBAtoGREY(const std::string &filename) {
	uchar4 * h_rgbaImage;
	uchar4 *d_rgbaImage;
	unsigned char * h_greyImage;
	unsigned char * d_greyImage;
		
	cv::Mat image; 
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		printf("oy , it's empty");
		getchar();
		exit(1);
	}
	
	cv::Mat imageRGBA;
	cv::Mat imageGrey;

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	imageGrey.create(image.rows, image.cols, CV_8UC1);

	h_rgbaImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);

	const size_t numPixels = imageRGBA.rows * imageRGBA.cols;
	cudaMalloc(&d_rgbaImage, sizeof(uchar4)* numPixels);
	cudaMalloc(&d_greyImage, sizeof(char) * numPixels);
	
	cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice); 
	
	dim3 threadBlocks(THREAD_BLOCK_DIM, THREAD_BLOCK_DIM);
	dim3 gridBlocks((imageRGBA.cols / threadBlocks.y) + 1, (imageRGBA.rows/ threadBlocks.x) + 1);

	std::cout << "THREAD blocks dimensions : " << threadBlocks.x << " x " << threadBlocks.y << std::endl;
	std::cout << "GRID blocks dimensions : " << gridBlocks.x << " x " << gridBlocks.y << std::endl;

	rgbaToGrey <<< gridBlocks, threadBlocks >>> (d_rgbaImage, d_greyImage,imageRGBA.rows , imageRGBA.cols); 
	
	h_greyImage = new unsigned char[numPixels];
	cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels,cudaMemcpyDeviceToHost);

	cv::Mat gray = cv::Mat(imageRGBA.rows , imageRGBA.cols, CV_8UC1, h_greyImage);
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv:imshow("Greyed image", gray);
	cv::waitKey(0);
}

int main(int argc, char** argv) {
	std::string fileName = "Image.jpg";
	runRGBAtoGREY(fileName);
	return 0;
}