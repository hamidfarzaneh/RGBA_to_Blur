
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "GpuTimer.h"

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

void runRGBAtoGREY(const std::string &filename,const std::string &outfilename) {
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
	GPUTimer timer;
	timer.start_timer();
	rgbaToGrey <<< gridBlocks, threadBlocks >>> (d_rgbaImage, d_greyImage,imageRGBA.rows , imageRGBA.cols); 
	timer.stop_timer();
	printf("simple cuda code to convert : ");
	timer.print_elapsed_time();
	h_greyImage = new unsigned char[numPixels];
	cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels,cudaMemcpyDeviceToHost);

	cv::Mat gray = cv::Mat(imageRGBA.rows , imageRGBA.cols, CV_8UC1, h_greyImage);
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imwrite(outfilename , gray);
	//cv::waitKey(0);
}


/* Process USING OPENCV .
The image is loaded from the disk and the method cv::cvtColor from OpenCV4Tegra is used to convert the image into grayscale.
Included from :
http://www.coldvision.io/2015/11/17/rgb-grayscale-conversion-cuda-opencv/

*/
void processUsingOpenCV(std::string input_file, std::string output_file) {
	cv::Mat image;
	cv::Mat imageRGBA;
	cv::Mat imageGrey;
	image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << input_file << std::endl;
		exit(1);
	}

	GPUTimer timer;
	timer.start_timer();
	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);  // CV_BGR2GRAY

												  //allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	timer.stop_timer();

	int err = printf("OpenCV code ran in: ");
	timer.print_elapsed_time();

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	//output the image
	cv::imwrite(output_file.c_str(), imageGrey);
}


int main(int argc, char** argv) {
	std::string inputFileName = "Image.jpg";
	std::string outputFileName = "outOPENCV.jpg";
	std::string outputFileName2 = "outGPUSIMPLE.jpg";
	processUsingOpenCV(inputFileName, outputFileName);
	runRGBAtoGREY(inputFileName,outputFileName2);
	getchar();
	return 0;
}