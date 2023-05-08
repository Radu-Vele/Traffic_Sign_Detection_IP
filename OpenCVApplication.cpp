#include "stdafx.h"
#include "common.h"

#define MEM_ALLOC_ERR "Error: Failed to allocate memory on the heap.\n"
#define MEM_ERR_CODE -1

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

bool isMaxOnDir(int curr_row, int curr_col, Mat magnitude, int angle) {
	int curr_magnitude = magnitude.at<float>(curr_row, curr_col);
	if (angle < 0) {
		angle += 360;
	}
	if ((angle >= 22 && angle < 67) || (angle >= 202 && angle < 247)) { // 1
		return curr_magnitude >= magnitude.at<float>(curr_row + 1, curr_col - 1) &&
			(curr_magnitude >= magnitude.at<float>(curr_row - 1, curr_col + 1));
	}
	else if ((angle >= 67 && angle < 112) || (angle >= 247 && angle < 292)) { // 0
		return curr_magnitude >= magnitude.at<float>(curr_row - 1, curr_col) &&
			(curr_magnitude >= magnitude.at<float>(curr_row + 1, curr_col));
	}
	else if ((angle >= 112 && angle < 157) || (angle >= 292 && angle < 337)) { // 3
		return curr_magnitude >= magnitude.at<float>(curr_row - 1, curr_col - 1) &&
			(curr_magnitude >= magnitude.at<float>(curr_row + 1, curr_col + 1));
	}
	else { // 2
		return curr_magnitude >= magnitude.at<float>(curr_row, curr_col - 1) &&
			(curr_magnitude >= magnitude.at<float>(curr_row, curr_col + 1));
	}
}

int* computeHistogram(Mat src) {
	int rows = src.rows;
	int cols = src.cols;

	int* histogram = (int*)calloc(256, sizeof(int));
	if (!histogram) {
		printf(MEM_ALLOC_ERR);
		exit(MEM_ERR_CODE);
	}

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			float pixel_value = src.at<float>(i, j);
			int index = (int) pixel_value;
			histogram[index]++;
		}
	}

	return histogram;
}

Mat cannyEdgeDetection(Mat src) {
	int rows = src.rows;
	int cols = src.cols;

	Mat magnitude(rows, cols, CV_32FC1); // buffer for computed magnitude
	Mat angle(rows, cols, CV_32FC1); // buffer for computed angles
	Mat magnitude_max(rows, cols, CV_32FC1, Scalar(0)); // buffer for the maximum magnitudes on edge
	Mat scaled_magnitude_max;
	Mat scaled_magnitude_max_adaptive(rows, cols, CV_8UC1, Scalar(0));
	
	//blur input image
	Mat src_blurred = gaussianBlur(src);
	imshow("Blurred", src_blurred);	

	//kernels
	double sobel_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	double sobel_y[9] = { 1, 2, 1, 0, 0 , 0, -1, -2, -1 };

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			double delta_x = computeConvolution(src_blurred, i, j, sobel_x);
			double delta_y = computeConvolution(src_blurred, i, j, sobel_y); //TODO: Shouldn't we apply the vertical convolution on the intermmediate image obtained after applying the horizontal convolution? I don't think so
			magnitude.at<float>(i, j) = (float) sqrt(pow(delta_x, 2) + pow(delta_y, 2));
			angle.at<float>(i, j) = (float)(atan2(delta_y, delta_x) * (180.0 / PI));
		}
	}
	
	//Non-maxima suppression
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols -1; j++) {
			if (isMaxOnDir(i, j, magnitude, angle.at<float>(i, j))) {
				magnitude_max.at<float>(i, j) = magnitude.at<float>(i, j);
			}
		}
	}


	//Adaptive thresholding
	float sobel_scaling_factor = 1.0 / (4.0 * sqrt(2));
	scaled_magnitude_max = sobel_scaling_factor * magnitude_max;

	int* magnitude_histogram_scaled = computeHistogram(scaled_magnitude_max);

	float p_value = 0.45; // set value between 0.01 and 0.1 ??
	int nr_no_edge_pixels = (int)((1 - p_value) * (float)((rows - 2) * (cols - 2) - magnitude_histogram_scaled[0]));

	int hist_count = 0;
	int adaptive_threshold = 0;
	for (adaptive_threshold = 1; adaptive_threshold < 256; adaptive_threshold++) {
		hist_count += magnitude_histogram_scaled[adaptive_threshold];
		if (hist_count >= nr_no_edge_pixels) {
			break;
		}
	}
	
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			if (scaled_magnitude_max.at<float>(i, j) > adaptive_threshold) { // or magnitude_max?
				scaled_magnitude_max_adaptive.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("adaptive", scaled_magnitude_max_adaptive);
	free(magnitude_histogram_scaled);

	//Weak edge removal

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