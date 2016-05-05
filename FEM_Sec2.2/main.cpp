#include <iostream>
#include <opencv/cv.h>

/**
 * 剛性マトリックスと線型方程式の右辺を作成する。
 *
 * @param A		剛性マトリックス（4x4の正方対象行列）
 * @param B		線型方程式の右辺
 * @param N		マトリックスのサイズ-1
 */
void stiff1(cv::Mat_<double>& A, cv::Mat_<double>& B, int N) {
	A = cv::Mat_<double>(N + 1, N + 1, 0.0);
	B = cv::Mat_<double>(N + 1, 1, 0.0);

	double x = 1.0 / (double)N;

	for (int i = 0; i < N; ++i) {
		A(i, i) += 1.0 / x;
		A(i + 1, i + 1) += 1.0 / x;
		A(i, i + 1) -= 1.0 / x;
		A(i + 1, i) -= 1.0 / x;

		B(i, 0) += x / 2.0;
		B(i + 1, 0) += x / 2.0;
	}
}

/**
 * 境界条件
 * u1=u_N=0
 */
void bc1(int N, std::vector<int>& given_indices) {
	given_indices.push_back(0);
	given_indices.push_back(N);
}

/**
 * 解析的に解いた答えを表示する
 */
void check_solution(cv::Mat_<double>& B, int N) {
	printf(" FEM result | True value\n");
	printf("------------+------------\n");
	for (int i = 0; i <= N; ++i) {
		double x = (double)i / (double)N;
		printf(" %10.3E | %10.3E\n", B(i, 0), 0.5*(x - x*x));
	}
}

/**
* 線型方程式が正則となるよう、式変形する。
*/
void bound2(cv::Mat_<double>& A, cv::Mat_<double>& C, std::vector<int>& given_indices, std::vector<int>& given_nonzero_indices, std::vector<double>& given_nonzero_values) {
	for (int i = 0; i < given_nonzero_indices.size(); ++i) {
		for (int k = 0; k < A.rows; ++k) {
			C(k, 0) -= given_nonzero_values[i] * A(k, given_nonzero_indices[i]);
		}
	}

	for (int i = 0; i < given_indices.size(); ++i) {
		C(given_indices[i], 0) = 0.0;
	}
	for (int i = 0; i < given_nonzero_indices.size(); ++i) {
		C(given_nonzero_indices[i], 0) = given_nonzero_values[i];
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
	const int N = 3;

	cv::Mat_<double> A;
	cv::Mat_<double> B;

	stiff1(A, B, N);

	std::vector<int> given_indices;
	std::vector<int> given_nonzero_indices;
	std::vector<double> given_nonzero_values;

	bc1(N, given_indices);

	bound2(A, B, given_indices, given_nonzero_indices, given_nonzero_values);
	
	cv::Mat_<double> U = A.inv() * B;

	check_solution(U, N);

	return 0;
}