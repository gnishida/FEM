#include <iostream>
#include <fstream>
#include <opencv/cv.h>

using namespace std;

void datain3(int& N, vector<double>& coords, vector<pair<int, int>>& lnods, vector<int>& given_indices, vector<int>& given_nonzero_indices, vector<double>& given_nonzero_values, int icase) {
	ifstream in;
	if (icase == 1) {
		in.open("case1.dat");
	}
	else if (icase == 2) {
		in.open("case2.dat");
	}
	in >> N;

	// read coords
	for (int i = 0; i < N; ++i) {
		int node_id;
		int given;
		double coord;
		in >> node_id >> given >> coord;
		if (given == 1) {
			given_indices.push_back(node_id - 1);
		}
		coords.push_back(coord);
	}

	in >> N;
	for (int i = 0; i < N; ++i) {
		int node_id;
		int idx1, idx2;
		in >> node_id >> idx1 >> idx2;
		lnods.push_back(make_pair(idx1 - 1, idx2 - 1));
	}

	int num;
	in >> num;
	for (int i = 0; i < num; ++i) {
		int line_no;
		int node_id;
		double value;
		in >> line_no >> node_id >> value;
		given_nonzero_indices.push_back(node_id - 1);
		given_nonzero_values.push_back(value);
	}
}

/**
* 剛性マトリックスと線型方程式の右辺を作成する。
*
* @param A		剛性マトリックス（4x4の正方対象行列）
* @param B		線型方程式の右辺
* @param N		マトリックスのサイズ-1
*/
void stiff4(cv::Mat_<double>& A, cv::Mat_<double>& B, int N, vector<double> coords, vector<pair<int, int>> lnods) {
	A = cv::Mat_<double>(N + 1, N + 1, 0.0);
	B = cv::Mat_<double>(N + 1, 1, 0.0);

	cv::Mat_<double> astiff = cv::Mat_<double>(2, 2, 0.0);
	cv::Mat_<double> c = cv::Mat_<double>(2, 1, 0.0);

	for (int i = 0; i < N; ++i) {
		double x = coords[lnods[i].second] - coords[lnods[i].first];

		// 要素マトリックスを作成
		astiff(0, 0) = 1.0 / x;
		astiff(1, 0) = -1.0 / x;
		astiff(0, 1) = -1.0 / x;
		astiff(1, 1) = 1.0 / x;
		c(0, 0) = x / 2.0;
		c(1, 0) = x / 2.0;

		int ip1 = lnods[i].first;
		int ip2 = lnods[i].second;

		// 全体マトリックスへマージする
		A(ip1, ip1) += astiff(0, 0);
		A(ip2, ip2) += astiff(1, 1);
		A(ip1, ip2) += astiff(0, 1);
		A(ip2, ip1) += astiff(1, 0);
		B(ip1, 0) += c(0, 0);
		B(ip2, 0) += c(1, 0);
	}
}

/**
* 境界条件
* u1=u_N=0
*/
void bc2(int N, vector<int>& given_indices) {
	given_indices.push_back(0);
	given_indices.push_back(N);
}

/**
* 解析的に解いた答えを表示する
*/
void check_solution3(cv::Mat_<double>& B, int N, vector<double> coords, int icase) {
	printf(" FEM result | True value\n");
	printf("------------+------------\n");
	for (int i = 0; i <= N; ++i) {
		double x = coords[i];
		double u;

		if (icase == 1) {
			u = -0.5 * x*x - 0.5 * x + 1.0;
		}
		else if (icase == 2) {
			u = -0.5 * x*x + 1.5 * x + 1.0;
		}
		printf(" %10.3E | %10.3E\n", B(i, 0), u);
	}
}

/**
* 線型方程式が正則となるよう、式変形する。
*/
void bound2(cv::Mat_<double>& A, cv::Mat_<double>& C, vector<int>& given_indices, vector<int>& given_nonzero_indices, vector<double>& given_nonzero_values) {
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
	const int icase = 2;

	int N;
	vector<double> coords;
	vector<pair<int, int>> lnods;
	vector<int> given_indices;
	vector<int> given_nonzero_indices;
	vector<double> given_nonzero_values;

	datain3(N, coords, lnods, given_indices, given_nonzero_indices, given_nonzero_values, icase);

	cv::Mat_<double> A;
	cv::Mat_<double> B;

	stiff4(A, B, N, coords, lnods);

	bc2(N, given_indices);

	bound2(A, B, given_indices, given_nonzero_indices, given_nonzero_values);

	cv::Mat_<double> U = A.inv() * B;

	check_solution3(U, N, coords, icase);

	return 0;
}