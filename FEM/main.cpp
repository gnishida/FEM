#include <iostream>
#include <opencv/cv.h>

/**
 * 剛性マトリックスを作成する
 */
void stiff(cv::Mat_<double>& A, int N, double k) {
	A = cv::Mat_<double>(N + 1, N + 1, 0.0);

	for (int i = 0; i < N; ++i) {
		A(i, i) += k;
		A(i, i + 1) -= k;
		A(i + 1, i) -= k;
		A(i + 1, i + 1) += k;
	}
}

void ex1(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);
	C(N, 0) = 1.0;

	given_indices.push_back(0);

	given_values.push_back(0);
}

void ex2(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);
	C(2, 0) = 1.0;

	given_indices.push_back(0);
	given_indices.push_back(4);

	given_values.push_back(0);
	given_values.push_back(0);
}

void ex3(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);

	given_indices.push_back(0);
	given_indices.push_back(2);
	given_indices.push_back(4);

	given_values.push_back(0);
	given_values.push_back(1);
	given_values.push_back(0);
}

void bound(cv::Mat_<double>& A, cv::Mat_<double>& C, std::vector<int>& given_indices, std::vector<double>& given_values) {
	for (int i = 0; i < given_indices.size(); ++i) {
		for (int k = 0; k < A.rows; ++k) {
			C(k, 0) -= given_values[i] * A(k, given_indices[i]);
		}
	}

	for (int i = 0; i < given_indices.size(); ++i) {
		C(given_indices[i], 0) = given_values[i];
	}

	for (int i = 0; i < given_indices.size(); ++i) {
		for (int k = 0; k < A.rows; ++k) {
			A(k, given_indices[i]) = 0.0;
			A(given_indices[i], k) = 0.0;
		}
		A(given_indices[i], given_indices[i]) = 1.0;
	}
}

int main() {
	const int N = 4;

	cv::Mat_<double> A;
	cv::Mat_<double> C;

	double k = 1;
	
	stiff(A, N, k);
	std::cout << "A:" << std::endl << A << std::endl;

	std::vector<int> given_indices;
	std::vector<double> given_values;
	//ex1(C, N, given_indices, given_values);
	ex2(C, N, given_indices, given_values);
	//ex3(C, N, given_indices, given_values);
	std::cout << "C:" << std::endl << C << std::endl;

	bound(A, C, given_indices, given_values);
	std::cout << "A:" << std::endl << A << std::endl;

	cv::Mat_<double> B = A.inv() * C;
	std::cout << "B:" << std::endl << B << std::endl;

	return 0;
}