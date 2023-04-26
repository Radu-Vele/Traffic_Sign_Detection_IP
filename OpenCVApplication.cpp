#include "stdafx.h"
#include "common.h"

double computeConvolution(Mat src, int i, int j, double* kernel);

Mat colorToGrayscale(Mat src) {
	int rows = src.rows;
	int cols = src.cols;
	Mat dst(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Vec3b channels_val = src.at<Vec3b>(i, j);
			dst.at<uchar>(i, j) = (channels_val[0] + channels_val[1] + channels_val[2]) / 3;
		}
	}
	return dst;
}

Mat gaussianBlur(Mat src) {
	int rows = src.rows;
	int cols = src.cols;
	double gaussian_kernel[9] = { 1.0 / 16, 2.0 / 16, 1.0 / 16, 2.0 / 16, 4.0 / 16, 2.0 / 16, 1.0 / 16, 2.0 / 16, 1.0 / 16 };
	Mat dst(rows, cols, CV_8UC1);

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			dst.at<uchar>(i, j) = (uchar) computeConvolution(src, i, j, gaussian_kernel);
		}
	}

	return dst;
}

/*
* Use a 3x3 kernel by default
*/
double computeConvolution(Mat src, int i, int j, double* kernel) {
	double result;
	result = (double)src.at<uchar>(i - 1, j - 1) * kernel[0];
	result += (double)src.at<uchar>(i - 1, j) * kernel[1];
	result += (double)src.at<uchar>(i - 1, j + 1) * kernel[2];
	result += (double)src.at<uchar>(i, j - 1) * kernel[3];
	result += (double)src.at<uchar>(i, j) * kernel[4];
	result += (double)src.at<uchar>(i, j + 1) * kernel[5];
	result += (double)src.at<uchar>(i + 1, j - 1) * kernel[6];
	result += (double)src.at<uchar>(i + 1, j) * kernel[7];
	result += (double)src.at<uchar>(i + 1, j + 1) * kernel[8];
	return result;
}

Mat cannyEdgeDetection(Mat src) {
	int rows = src.rows;
	int cols = src.cols;

	Mat src_blurred = gaussianBlur(src);
	imshow("Blurred", src_blurred);	
	Mat magnitude(rows, cols, CV_32SC1);
	Mat angle(rows, cols, CV_32SC1);

	//kernels
	double sobel_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	double sobel_y[9] = { 1, 2, 1, 0, 0 , 0, -1, -2, -1 };

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			double delta_x = computeConvolution(src_blurred, i, j, sobel_x);
			double delta_y = computeConvolution(src_blurred, i, j, sobel_y);
			magnitude.at<int>(i, j) = (int) sqrt(pow(delta_x, 2) + pow(delta_y, 2));
			angle.at<int>(i, j) = (int)(atan2(delta_y, delta_x) * (180.0 / PI));
		}
	}

	//Non-maxima suppression
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols -1; j++) {
			printf("(%d, %d) ", magnitude.at<int>(i, j), angle.at<int>(i, j));
		}
		printf("\n");
	}

	return src_blurred;
}

void processInput() {
	Mat input_color = imread("./Images/harbor.bmp", IMREAD_COLOR);
	Mat input_gray = colorToGrayscale(input_color);
	imshow("Input Gray", input_gray);
	cannyEdgeDetection(input_gray);
	waitKey(0);
}

int main()
{
	processInput();
}