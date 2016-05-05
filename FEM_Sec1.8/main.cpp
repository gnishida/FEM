#include <iostream>
#include <opencv/cv.h>

/**
 * バネ釣り合いの剛性マトリックスを作成する
 *
 * @param A	剛性マトリックス（5x5の正方対象行列）
 * @param N 行列のサイズ-1
 * @param k バネ係数
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

/**
 * Ex1の境界条件をセットする。
 * f5 = 1なので、C(5, 0) = 1とセットする。
 * また、u1=0なので、既知のインデックスは0、値は0をセットする。
 *
 * @param C				線型方程式の右辺（5x1の列ベクトル）
 * @param N				未知数の数-1
 * @param given_indices	既知のu値が入るインデックス
 * @param given_values	既知のu値
 */
void ex1(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);
	C(N, 0) = 1.0;

	given_indices.push_back(0);

	given_values.push_back(0);
}

/**
* Ex2の境界条件をセットする。
* f3 = 1なので、C(2, 0) = 1とセットする。
* また、u1=u5=0なので、既知のインデックスは0と4、値は0と0をセットする。
*
* @param C				線型方程式の右辺（5x1の列ベクトル）
* @param N				未知数の数-1
* @param given_indices	既知のu値が入るインデックス
* @param given_values	既知のu値
*/
void ex2(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);
	C(2, 0) = 1.0;

	given_indices.push_back(0);
	given_indices.push_back(4);

	given_values.push_back(0);
	given_values.push_back(0);
}

/**
* Ex3の境界条件をセットする。
* u1=u5=0、u3=1なので、既知のインデックスは0、2、4、値は0、1、0をセットする。
*
* @param C				線型方程式の右辺（5x1の列ベクトル）
* @param N				未知数の数-1
* @param given_indices	既知のu値が入るインデックス
* @param given_values	既知のu値
*/
void ex3(cv::Mat_<double>& C, int N, std::vector<int>& given_indices, std::vector<double>& given_values) {
	C = cv::Mat_<double>(N + 1, 1, 0.0);

	given_indices.push_back(0);
	given_indices.push_back(2);
	given_indices.push_back(4);

	given_values.push_back(0);
	given_values.push_back(1);
	given_values.push_back(0);
}

/**
 * 線型方程式が正則となるよう、式変形する。
 */
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