
#include "cuda_runtime.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define ROWS 5
#define COLUMNS 5

using namespace std;
using namespace cv;

void convolution(unsigned char*, float*, int, int, int);
__global__ void kernel(unsigned char*, float*, int, int, int);

int main() {
	Mat Image = imread("C:\\Users\\Stella\\Documents\\Visual Studio 2015\\Projects\\nra\\flower.png", IMREAD_COLOR);
	int height = Image.rows;
	int width = Image.cols;
	int channels = Image.channels();

	float filter[ROWS * COLUMNS] = {
		0.04, 0.04, 0.04, 0.04, 0.04,
		0.04, 0.04, 0.04, 0.04, 0.04,
		0.04, 0.04, 0.04, 0.04, 0.04,
		0.04, 0.04, 0.04, 0.04, 0.04,
		0.04, 0.04, 0.04, 0.04, 0.04
	};
	
	convolution(Image.data, filter, height, width, channels);

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", Image);
	waitKey(0);

	return 0;
}

void convolution(unsigned char* img, float* filter, int H, int W, int C) {
	unsigned char* dev_img = NULL;
	float* dev_filter = NULL;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dev_img, H * W * C);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_img failed!");
		cudaFree(dev_img);
		cudaFree(dev_filter);
	}

	cudaStatus = cudaMalloc((void**)&dev_filter, ROWS * COLUMNS * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_input failed!");
		cudaFree(dev_img);
		cudaFree(dev_filter);
	}

	cudaStatus = cudaMemcpy(dev_img, img, H * W * C, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyHostToDevice dev_img failed!");
		cudaFree(dev_img);
		cudaFree(dev_filter);
	}

	cudaStatus = cudaMemcpy(dev_filter, filter, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyHostToDevice dev_filter failed!");
		cudaFree(dev_img);
		cudaFree(dev_filter);
	}

	dim3 dimGrid(ceil((float)W / 16), ceil((float)W / 16));
	dim3 dimBlock(16, 16, 1);

	kernel<<<dimGrid, dimBlock>>>(dev_img, dev_filter, H, W, C);

	cudaStatus = cudaMemcpy(img, dev_img, H * W * C, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyDeviceToHost img failed!");
		cudaFree(dev_img);
		cudaFree(dev_filter);
	}

	cudaFree(dev_img);
	cudaFree(dev_filter);
}

__global__ void kernel(unsigned char* img, float* filter, int H, int W, int C) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int rowsRadius = ROWS / 2;
	int colsRadius = COLUMNS / 2;
	float accum = 0;

	for (int i = 0; i < C; i++) {
		if (row < H && col < W) {
			int startRow = row - rowsRadius;
			int startCol = col - colsRadius;

			for (int j = 0; j < ROWS; j++) {
				for (int k = 0; k < COLUMNS; k++) {
					int currRow = startRow + i;
					int currCol = startCol + j;

					if (currRow >= 0 && currRow < H && currCol >= 0 && currCol < W) {

						accum += img[(currRow * W + currCol) * C + i] * filter[j * ROWS + k];
					}
					else accum = 0;
				}
			}
			img[(row * W + col) * C + i] = accum;
		}
	}
}
