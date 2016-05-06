/**
 * テキスト p. 157
 * バージョン6  2次の補間関数
 */

#include <iostream>
#include <fstream>
#include <opencv/cv.h>

using namespace std;

void datain6(int& nnode, int& nelem, vector<double>& coords, vector<vector<int>>& lnods, vector<int>& given_indices, vector<int>& given_nonzero_indices, vector<double>& given_nonzero_values, int icase, int& nint, vector<int>& ntnoel) {
	ifstream in;
	if (icase == 11) {
		in.open("case11.dat");
	}
	else if (icase == 12) {
		in.open("case12.dat");
	}
	else if (icase == 21) {
		in.open("case21.dat");
	}
	else if (icase == 22) {
		in.open("case22.dat");
	}
	in >> nnode;

	// read coords
	coords.resize(nnode);
	for (int i = 0; i < nnode; ++i) {
		int node_id;
		int given;

		in >> node_id >> given >> coords[i];
		if (given == 1) {
			given_indices.push_back(node_id - 1);
		}
	}

	in >> nelem;
	lnods.resize(nelem);
	ntnoel.resize(nelem);
	for (int ielem = 0; ielem < nelem; ++ielem) {
		int node_id;
		in >> node_id >> ntnoel[ielem];

		for (int i = 0; i < ntnoel[ielem]; ++i) {
			int idx;
			in >> idx;
			lnods[ielem].push_back(idx - 1);
		}
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

	in >> nint;
}

/**
* 要素マトリックスを作成する。
*
* @param coords	各ノードの座標
* @param lnods	ノードの接続（lnods[0]とlnods[1]）を表す
* @param astiff	要素マトリックス
* @param nint	数値積分の区分数
*/
void element(vector<double> coords, vector<int> lnods, cv::Mat_<double>& astiff, int nint, int ntnoel) {
	double gsp[][4] = {
			{ 0.0, 0.0, 0.0, 0.0 },
			{ -0.577350269189626, 0.577350269189626, 0.0, 0.0 },
			{ -0.774596669241483, 0.0, 0.774596669241483, 0.0 },
			{ -0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053 }
	};
	double wgh[][4] = {
			{ 2.0, 0.0, 0.0, 0.0 },
			{ 1.0, 1.0, 0.0, 0.0 },
			{ 0.555555555555556, 0.888888888888889, 0.555555555555556, 0.0 },
			{ 0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454 }
	};

	astiff = cv::Mat_<double>(ntnoel, ntnoel, 0.0);

	for (int ir = 0; ir < nint; ++ir) {
		double r = gsp[nint - 1][ir];

		vector<double> dndr(ntnoel);
		if (ntnoel == 2) {
			dndr[0] = -0.5;
			dndr[1] = 0.5;
		}
		else if (ntnoel == 3) {
			dndr[0] = r - 0.5;
			dndr[1] = r + 0.5;
			dndr[2] = -2.0 * r;
		}
		
		double ajacob = 0.0;
		
		for (int i = 0; i < ntnoel; ++i) {
			ajacob += dndr[i] * coords[lnods[i]];
		}
		double detjac = ajacob;
		double ajainv = 1.0 / ajacob;

		vector<double> dndx(ntnoel);
		for (int i = 0; i < ntnoel; ++i) {
			dndx[i] = dndr[i] * ajainv;
		}

		double detwei = detjac * wgh[nint - 1][ir];

		for (int i = 0; i < ntnoel; ++i) {
			for (int j = 0; j < ntnoel; ++j) {
				astiff(j, i) += detwei * dndx[i] * dndx[j];
			}
		}
	}
}

/**
 * 要素マトリックスを全体マトリックスへマージする
 *
 * @param A			全体マトリックス
 * @param lnods		ノードの接続
 * @param astiff	要素マトリックス
 * @param ntnoel	区間のノード数
 */
void merge(cv::Mat_<double>& A, vector<int> lnods, cv::Mat_<double> astiff, int ntnoel) {
	vector<int> ip(ntnoel);

	for (int i = 0; i < ntnoel; ++i) {
		ip[i] = lnods[i];
	}

	for (int j = 0; j < ntnoel; ++j) {
		for (int i = 0; i < ntnoel; ++i) {
			A(ip[i], ip[j]) += astiff(i, j);
		}
	}
}

/**
* 剛性マトリックスと線型方程式の右辺を作成する。
*
* @param A		剛性マトリックス（4x4の正方対象行列）
* @param B		線型方程式の右辺
* @param nnode	マトリックスのサイズ
* @param coords	ノードの座標
* @param nelem	要素の数
* @param lnods	要素の接続を表す
* @param nint	数値積分のためのサンプリング点数
* @param ntnoel	各区間のノード数
*/
void stiff6(cv::Mat_<double>& A, cv::Mat_<double>& B, int nnode, vector<double> coords, int nelem, vector<vector<int>> lnods, int nint, vector<int> ntnoel) {
	A = cv::Mat_<double>(nnode, nnode, 0.0);
	B = cv::Mat_<double>(nnode, 1, 0.0);

	cv::Mat_<double> astiff = cv::Mat_<double>(2, 2, 0.0);
	cv::Mat_<double> c;

	for (int ielem = 0; ielem < nelem; ++ielem) {
		// 要素マトリックスを作成
		element(coords, lnods[ielem], astiff, nint, ntnoel[ielem]);

		// 全体マトリックスへマージ
		merge(A, lnods[ielem], astiff, ntnoel[ielem]);

		vector<int> ip(ntnoel[ielem]);
		for (int i = 0; i < ntnoel[ielem]; ++i) {
			ip[i] = lnods[ielem][i];
		}

		double x = coords[lnods[ielem][1]] - coords[lnods[ielem][0]];
		if (ntnoel[ielem] == 2) {		// 1次補間
			c = cv::Mat_<double>(2, 1, 0.0);
			c(0, 0) = x / 2.0;
			c(1, 0) = x / 2.0;
		}
		else if (ntnoel[ielem] == 3) {	// 2次補間
			c = cv::Mat_<double>(3, 1, 0.0);
			c(0, 0) = x / 6.0;
			c(1, 0) = x / 6.0;
			c(2, 0) = x * 2.0 / 3.0;
		}

		for (int i = 0; i < ntnoel[ielem]; ++i) {
			B(ip[i], 0) += c(i, 0);
		}
	}
}

/**
* 解析的に解いた答えを表示する
*/
void check_solution6(cv::Mat_<double>& B, vector<double> coords, int icase) {
	printf(" FEM result | True value\n");
	printf("------------+------------\n");
	for (int i = 0; i < B.rows; ++i) {
		double x = coords[i];
		double u;

		if (icase < 20) {
			u = -0.5 * x*x - 0.5 * x + 1.0;
		}
		else if (icase > 20) {
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
	const int icase = 22;

	int nnode;
	int nelem;
	vector<double> coords;
	vector<vector<int>> lnods;
	vector<int> given_indices;
	vector<int> given_nonzero_indices;
	vector<double> given_nonzero_values;
	int nint;
	vector<int> ntnoel;

	datain6(nnode, nelem, coords, lnods, given_indices, given_nonzero_indices, given_nonzero_values, icase, nint, ntnoel);

	cv::Mat_<double> A;
	cv::Mat_<double> B;

	stiff6(A, B, nnode, coords, nelem, lnods, nint, ntnoel);

	bound2(A, B, given_indices, given_nonzero_indices, given_nonzero_values);

	cv::Mat_<double> U = A.inv() * B;

	check_solution6(U, coords, icase);

	return 0;
}