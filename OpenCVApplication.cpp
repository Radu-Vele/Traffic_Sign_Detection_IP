#include "stdafx.h"
#include "common.h"
#include <queue>
#include <unordered_map>

using namespace std;
using namespace cv;

#define MEM_ALLOC_ERR "Error: Failed to allocate memory on the heap.\n"
#define MEM_ERR_CODE -1


int n8_di[8] = { 0,-1,-1, -1, 0, 1, 1, 1 };
int n8_dj[8] = { 1, 1, 0, -1, -1,-1, 0, 1 };

vector<double> circle_signature;
double th_circle = 3;

vector<double> triangle_signature;
double th_triangle = 9;

vector<double> square_signature;
double th_square = 3;

//Mat input_color = imread("./Images/harbor.bmp", IMREAD_COLOR);
//Mat input_color = imread("./Images/test1.bmp", IMREAD_COLOR);
//Mat input_color = imread("./Images/test2.bmp", IMREAD_COLOR);
Mat input_color = imread("./Images/test4.bmp", IMREAD_COLOR);

typedef struct contour {
	vector<Point> border;
	vector<int> dir_vector;
	bool loop;
	int size;
};

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
	Mat dst(rows, cols, CV_8UC1, Scalar(0));

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			dst.at<uchar>(i, j) = (uchar)computeConvolution(src, i, j, gaussian_kernel);
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
			int index = (int)pixel_value;
			histogram[index]++;
		}
	}

	return histogram;
}

/**
* Considers the image border as out of bounds
*/
bool outOfBounds(int curr_i, int curr_j, int rows, int cols) {
	if (curr_i >= rows - 1 || curr_i < 1) {
		return true;
	}
	if (curr_j >= cols - 1 || curr_j < 1) {
		return true;
	}
	return false;
}


Mat cannyEdgeDetection(Mat src) {
	int rows = src.rows;
	int cols = src.cols;

	Mat magnitude(rows, cols, CV_32FC1, Scalar(0)); // buffer for computed magnitude
	Mat angle(rows, cols, CV_32FC1, Scalar(0)); // buffer for computed angles
	Mat magnitude_max(rows, cols, CV_32FC1, Scalar(0)); // buffer for the maximum magnitudes on edge
	Mat scaled_magnitude_max;
	Mat scaled_magnitude_th(rows, cols, CV_8UC1, Scalar(0));

	//blur input
	Mat src_blurred = gaussianBlur(src);

	//kernels
	double sobel_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	double sobel_y[9] = { 1, 2, 1, 0, 0 , 0, -1, -2, -1 };

	for (int i = 2; i < rows - 2; i++) {
		for (int j = 2; j < cols - 2; j++) {
			double delta_x = computeConvolution(src_blurred, i, j, sobel_x);
			double delta_y = computeConvolution(src_blurred, i, j, sobel_y);
			magnitude.at<float>(i, j) = (float)sqrt(pow(delta_x, 2) + pow(delta_y, 2));
			angle.at<float>(i, j) = (float)(atan2(delta_y, delta_x) * (180.0 / PI));
		}
	}

	//Non-maxima suppression
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			if (isMaxOnDir(i, j, magnitude, angle.at<float>(i, j))) {
				magnitude_max.at<float>(i, j) = magnitude.at<float>(i, j);
			}
		}
	}


	//Adaptive thresholding
	float sobel_scaling_factor = 1.0 / (4.0 * sqrt(2));
	scaled_magnitude_max = sobel_scaling_factor * magnitude_max;

	int* magnitude_histogram_scaled = computeHistogram(scaled_magnitude_max);

	float p_value = 0.1;
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
			scaled_magnitude_th.at<uchar>(i, j) = (uchar)scaled_magnitude_max.at<float>(i, j);
		}
	}

	free(magnitude_histogram_scaled);

	//Weak edge removal
	float k = 0.4;

	//Label matrix elements as strong (255), weak (128) or no edge (0)
	int threshold_low = (int)(k * (float)adaptive_threshold);

	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			uchar curr_val = scaled_magnitude_th.at<uchar>(i, j);

			if (curr_val < threshold_low) { // no edge
				scaled_magnitude_th.at<uchar>(i, j) = 0;
			}
			else if (curr_val < adaptive_threshold) { // weak edge
				scaled_magnitude_th.at<uchar>(i, j) = 128;
			}
			else { // strong edge
				scaled_magnitude_th.at<uchar>(i, j) = 255;
			}
		}
	}

	Mat visited_mask(rows, cols, CV_8UC1, Scalar(0)); //keep track of visited nodes (may be replaced with a map or so)
	int neigh_nr = 8;
	int offset_i[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int offset_j[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };



	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			if (scaled_magnitude_th.at<uchar>(i, j) == 255 && visited_mask.at<uchar>(i, j) != 255) { // found the first unvisited strong edge.
				std::queue<Point> queue; // empty queue for the bfs
				visited_mask.at<uchar>(i, j) = 255;
				queue.push(Point(i, j));

				while (!queue.empty()) {
					Point curr = queue.front();
					queue.pop();

					for (int k = 0; k < neigh_nr; k++) { //check neighboring endges
						int new_i = curr.x + offset_i[k];
						int new_j = curr.y + offset_j[k];

						if (!outOfBounds(new_i, new_j, rows, cols)) {
							//mark the weak edge points as strong edges
							if (scaled_magnitude_th.at<uchar>(new_i, new_j) == 128) {
								scaled_magnitude_th.at<uchar>(new_i, new_j) = 255;
								if (visited_mask.at<uchar>(new_i, new_j) == 0) { //not visited
									visited_mask.at<uchar>(new_i, new_j) = 255;
									queue.push(Point(new_i, new_j)); //add all strong (prv. weak) edge neighbors to the queue
								}
							}
							else if (scaled_magnitude_th.at<uchar>(new_i, new_j) == 255) {
								if (visited_mask.at<uchar>(new_i, new_j) == 0) { //not visited
									visited_mask.at<uchar>(new_i, new_j) = 255;
									queue.push(Point(new_i, new_j)); //add all strong edge neighbors to the queue
								}
							}
						}
					}
				}
			}
		}
	}

	//eliminate all remaining weak edges
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			if (scaled_magnitude_th.at<uchar>(i, j) == 128) {
				scaled_magnitude_th.at<uchar>(i, j) = 0;
			}
		}
	}

	return scaled_magnitude_th;
}


Point find_P_0(Mat source) {
	/*
	 * Find the initial point of the contour and return it
	 */
	Point P_0;

	for (int i = source.rows - 1; i >= 0; i--) {
		for (int j = source.cols - 1; j >= 0; j--) {
			if (source.at<uchar>(i, j) != 255) {
				P_0.x = i;
				P_0.y = j;
				break;
			}
		}
	}

	return P_0;
}

contour extractContour(Mat source, Point P_0) {
	int dir, next_dir, curr_dir, k, size = 1;
	Point P_current;
	std::vector<Point> border;
	std::vector<int> dir_vector;
	bool cont = true, found, loop = true;
	P_current = P_0;
	border.push_back(P_current);
	dir = 7;

	while (cont) {
		
		if (dir % 2 == 0) {
			next_dir = (dir + 7) % 8;
		}
		else {
			next_dir = (dir + 6) % 8;
		}

		k = 0;
		found = false;

		while (k < 8 && !found) {
			int new_i = P_current.x + n8_di[next_dir];
			int new_j = P_current.y + n8_dj[next_dir];
			if (source.at<uchar>(new_i, new_j) == 0) {
				P_current.y = new_j;
				P_current.x = new_i;
				border.push_back(P_current);
				dir_vector.push_back(next_dir);
				dir = next_dir;
				found = true;
				size++;
			}

			k++;
			next_dir = (next_dir + 1) % 8;
		}

		if (k == 8) {
			cont = false; 
			loop = false;
		}

		if (border.size() > 2) {
			for (int i = 0; i < border.size() - 1; i++) {
				if (border[border.size() - 1] == border[i])
					cont = false;
			}
		}
	}

		return { border, dir_vector, loop, size };
}

/*
* Draw the contour using the border variable from cnt structure
*/
Mat drawContour(contour cnt, Mat source) {
	Mat dst;
	dst = source.clone();
	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < source.cols; j++) {
			dst.at<uchar>(i, j) = 255;
		}
	}
	for (int i = 0; i < cnt.border.size(); i++) {
		dst.at<uchar>(cnt.border[i].x, cnt.border[i].y) = 0;
	}
	return dst;
}

Mat inverseColors(Mat src) {

	Mat result = src.clone();
	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}

	return result;
}

Point getCenter(Mat binary_object) {
	int rows = binary_object.rows;
	int cols = binary_object.cols;

	Point center_mass;
	long long x = 0, y = 0;
	int area = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (binary_object.at<uchar>(i, j) == 0) {
				x += j;
				y += i;
				area++;
			}
		}
	}
	center_mass.x = x / area;
	center_mass.y = y / area;

	return center_mass;
}

Mat display_center_of_mass(Point center_of_mass, Mat source) {
	Mat result;
	result = source.clone();
	circle(result, center_of_mass, 5, Scalar(0, 0, 0), 1);

	return result;
}

Mat deleteEdge(Mat src, Mat edge) {

	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (edge.at<uchar>(i, j) == 0) {
				src.at<uchar>(i, j) = 255;
			}
		}
	}

	return src;
}

vector<double> getSignature(contour cnt, Point center) {

	vector<double> signature;
	vector<Point> border = cnt.border;
	int size = cnt.size;
	int dist;

	//compute euclidean distance to center for each point in the border
	for (int i = 0; i < size; i++) {
		double x = (double) center.x - border[i].y; 
		double y = (double) center.y - border[i].x; 
		double partial = sqrt(x * x + y * y);

		dist = sqrt(x * x + y * y);
		signature.push_back(dist);
	}

	return signature;
}

void printFunction(const std::string& name, vector<double> values, const int  interval_length, const int img_height) {
	Mat imgHist(img_height, interval_length, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	vector<int> scaled_vals;
	int max_vals = 0;
	for (int i = 0; i < interval_length; i++) {
		scaled_vals.push_back((int) (values[i] * 100));
		if (scaled_vals[i] > max_vals) {
			max_vals = scaled_vals[i];
		}
	}

	double scale = 1.0;
	scale = (double)img_height / max_vals;
	int baseline = img_height - 1;

	for (int x = 0; x < interval_length; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(scaled_vals[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

bool detectCircle(vector<int> signature) {

	for (int i = 0; i < signature.size(); i++) {
		if (signature[i] < 60) {
			return false;
		}
	}
	return true;
}

void drawResult(Mat curr_result, Vec3b box_color, Point center, int max) {
	max = (int) (max * 0.10) + max;
	int x = center.x - max;
	int y = center.y - max;
	int width = max * 2;
	int height = max * 2;
	// our rectangle...
	cv::Rect rect(x, y, width, height);
	// and its top left corner...
	cv::Point pt1(x, y);
	// and its bottom right corner.
	cv::Point pt2(x + width, y + height);
	// These two calls...
	cv::rectangle(curr_result, pt1, pt2, box_color);
}

/*
* Have all histogram values between lower and upper;
*/
vector<double> rescaleArray(vector<double> input, double lower, double upper) {
	int hist_size = input.size();
	double sum = 0;
	double min = 0; //consider the min distance 0
	double max = -1;
	vector<double> normalized_hist;

	for (int i = 0; i < hist_size; i++) {
		if (input[i] > max) {
			max = input[i];
		}
	}

	for (int i = 0; i < hist_size; i++) {
		float scaled_value = (input[i] - min) * (upper - lower) / (max - min) + lower; //fit in the wanted interval
		normalized_hist.push_back(scaled_value);
	}

	return normalized_hist;
}

vector<double> compressSignature(vector<double> input, int wanted_size) {
	int size = input.size();
	double step_size = (double) size / wanted_size;
	vector<double> compressed;
	
	for (int i = 0; i < wanted_size; i++) {
		double curr_index = i * step_size;
		int leftIndex = floor(curr_index);
		int rightIndex = ceil(curr_index);

		double leftVal = input[leftIndex];
		double rightVal = input[rightIndex];

		double new_value = leftVal + (curr_index - leftIndex) * (rightVal - leftVal);
		compressed.push_back(new_value);
	}
	
	return compressed;
}

vector<double> getNormalizedSampledSignature(Mat src) {
	Point P_0 = find_P_0(src); //first point of the edge
	contour cnt = extractContour(src, P_0);
	Mat mat_cnt = drawContour(cnt, src);
	Point center = getCenter(mat_cnt);
	vector<double> signature = getSignature(cnt, center);
	vector<double> normalized_signature = rescaleArray(signature, 0.0, 1.0);
	vector<double> compressed_signature = compressSignature(normalized_signature, 100);

	return compressed_signature;
}

double getMaxElem(vector<double> input) {
	int size = input.size();
	int max_val = INT_MIN;

	for (int i = 0; i < size; i++) {
		if (input[i] > max_val) {
			max_val = input[i];
		}
	}

	return max_val;
}

bool matchSignatures(vector<double> legacy, vector<double> curr, double th_shape) {
	int signature_size = curr.size();
	double delta_sum = 0;
	for (int i = 0; i < signature_size; i++) {
		delta_sum += abs(legacy[i] - curr[i]);
	}
	return (delta_sum < th_shape);
}

Vec3b compareWithShapes(vector<double> curr_signature) {
	//check if circle
	if (matchSignatures(circle_signature, curr_signature, th_circle)) {
		printf("Detected circle \n");
		return Vec3b(0, 0, 255);
	}
	
	//check if triangle
	if (matchSignatures(triangle_signature, curr_signature, th_triangle)) {
		printf("Detected triangle \n");
		return Vec3b(0, 255, 0);
		return true;
	}
	
	//check if square
	if (matchSignatures(square_signature, curr_signature, th_square)) {
		printf("Detected square \n");
		return Vec3b(255, 0, 255);
	}

	return Vec3b(0, 0, 0);
}

Mat processEdge(Mat src, Mat curr_result) {
	Point P_0 = find_P_0(src);
	contour cnt = extractContour(src, P_0);
	Mat mat_cnt = drawContour(cnt, src);
	Point center = getCenter(mat_cnt);

	if (cnt.loop) { //process only shapes that have a loop and are larger than 100
		if (cnt.size > 100) {
			vector<double> signature = getSignature(cnt, center);
			int max_dist = (int)getMaxElem(signature);
			vector<double> normalized_signature = rescaleArray(signature, 0.0, 1.0);
			vector<double> compressed_signature = compressSignature(normalized_signature, 100);

			Vec3b color_result = compareWithShapes(compressed_signature);
			if (!(color_result == Vec3b(0, 0, 0))) { // a color is returned
				//enclose in bounding box colored w.r.t shape
				drawResult(curr_result, color_result, center, max_dist);
			}

			imshow("center", display_center_of_mass(center, mat_cnt));
			waitKey(0);
		}		
	}

	Mat dst = deleteEdge(src, mat_cnt);
	return dst;
}

bool isEmpty(Mat src) {

	int rows = src.rows;
	int cols = src.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (src.at<uchar>(i, j) != 255) {
				return false;
			}
		}
	}
	return true;
}

void computePerfectShapesSignatures() {
	Mat input_circle = imread("Images/perfect_circle.bmp", IMREAD_GRAYSCALE);
	circle_signature = getNormalizedSampledSignature(input_circle);
	
	Mat input_triangle = imread("Images/perfect_triangle.bmp", IMREAD_GRAYSCALE);
	triangle_signature = getNormalizedSampledSignature(input_triangle);
	
	Mat input_square = imread("Images/perfect_square.bmp", IMREAD_GRAYSCALE);
	square_signature = getNormalizedSampledSignature(input_square);
}

void processInput() {
	Mat input_gray = colorToGrayscale(input_color);
	imshow("Input Gray", input_gray);
	Mat detected_edges = cannyEdgeDetection(input_gray);
	Mat inverse = inverseColors(detected_edges);
	imshow("Edges inversed", inverse);
	waitKey(0);

	//process all edges (check if sign)
	bool empty = false;
	while (!empty) {
		Mat edge = processEdge(inverse, input_color);
		empty = isEmpty(edge);
	}

	imshow("Final result", input_color);
	waitKey(0);
}

int main() {
	computePerfectShapesSignatures();
	processInput();
}