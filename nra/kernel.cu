
#include "cuda_runtime.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define ROWS 5
#define COLUMNS 5

using namespace std;
using namespace cv;


// Laplacian
/*__constant__ float filter[ROWS * COLUMNS] = {
						-1, -1, -1, -1, -1,
						-1, -1, -1, -1, -1,
						-1, -1, 24, -1, -1,
						-1, -1, -1, -1, -1,
						-1, -1, -1, -1, -1
					};*/

// Laplacian of Gaussian
__constant__ float filter[ROWS * COLUMNS] = {
						0, 0, -1, 0, 0,
						0, -1, -2, -1, 0,
						-1, -2, 16, -2, -1,
						0, -1, -2, -1, 0,
						0, 0, -1, 0, 0
					};


void convolution(float3*, int, int, float3*);
__device__ float3 add(float3, float3);
__device__ float3 multiply(float, float3);
__global__ void kernel(float3*, int, int, float3*);


int main() {
	Mat Image = imread("flower.png", IMREAD_COLOR);

	if (Image.empty()) {
		cout << "Could not read the image" << endl;
		return -1;
	}

	Image.convertTo(Image, CV_32FC3);
	Image /= 255;
	int height = Image.rows;
	int width = Image.cols;

	Mat Result(height, width, Image.type());
	float3* input = (float3*)Image.ptr<float3>();
	float3* output = (float3*)Result.ptr<float3>();
	
	convolution(input, height, width, output);

	Result *= 255;
	Result.convertTo(Result, CV_8UC3);

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", Result);
	waitKey(0);

	return 0;
}


void convolution(float3* input, int H, int W, float3* output) {
	float3* dev_input = NULL;
	float3* dev_output = NULL;

	int size = H * W * sizeof(float3);

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStatus = cudaMalloc((void**)&dev_input, size*sizeof(float3));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_input failed!\n");

	cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(float3));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc dev_output failed!\n");

	cudaStatus = cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpyHostToDevice dev_input failed!\n");

	dim3 dimGrid(ceil((float)W / 16), ceil((float)H / 16));
	dim3 dimBlock(16, 16, 1);

	cudaEventRecord(start, 0);
	kernel<<<dimGrid, dimBlock>>>(dev_input, H, W, dev_output);
	cudaEventRecord(stop, 0);

	cudaStatus = cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpyDeviceToHost img failed!\n");

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "It took " << milliseconds << " milliseconds" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_input);
	cudaFree(dev_output);
}


__device__ float3 add(float3 vec1, float3 vec2) {
	return{ vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z };
}


__device__ float3 multiply(float num, float3 vec) {
	return{ num * vec.x, num * vec.y, num * vec.z };
}


__global__ void kernel(float3* dev_input, int H, int W, float3* dev_output) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int rowsRadius = ROWS / 2;
	int colsRadius = COLUMNS / 2;

	if (row < H && col < W) {
		int startRow = row - rowsRadius;
		int startCol = col - colsRadius;
		float3 temp = { 0.f, 0.f, 0.f };

		for (int i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLUMNS; j++) {
				int currRow = startRow + i;
				int currCol = startCol + j;

				if (currRow >= 0 && currRow < H && currCol >= 0 && currCol < W)
					temp = add(temp, multiply(filter[i * ROWS + j], dev_input[currRow * W + currCol]));
			}
		}
		dev_output[row * W + col] = temp;
	}
}
