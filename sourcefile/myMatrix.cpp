/*----------------------------------------------------------------\
@ Numerical Programming by Young-Keun Kim - Handong Global University

Author           : Lee Chan Keun
Created          : 26-03-2018
Modified         : 14-12-2023
Language/ver     : C++ in MSVS2019

Description      : myMatrix.cpp
----------------------------------------------------------------*/

#include "myMatrix.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#define vec(x) at[x][0]

/*===============================================================*/
/*          				 method  				             */
/*===============================================================*/

//print Matrix
void Matrix::print(char* name){
	printf("%s = \n",  name);
	for(int i = 0; i<rows; i++){
		for(int j = 0; j<cols; j++){
			printf("%15.6f\t", at[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*================================*/
/*           declare              */
/*================================*/

// Matrix();
Matrix::Matrix(){
	rows = 0;
	cols = 0;
}

// Matrix(unsigned int rows, unsigned int cols);
Matrix::Matrix(unsigned int rows, unsigned int cols):rows(rows),cols(cols){
	if (rows == 0 || cols == 0){
		printf("Error : Matrix size is wrong! : declare");
		return;
	}
		
	// 1. Allocate row array first
	this->at =(double**)malloc(sizeof(double*) * rows); 
	
	// 2. Then, allocate column 
	for (int i = 0; i < rows; i++)
		at[i] = (double*)malloc(sizeof(double) * cols);
}

// Matrix(unsigned int rows, unsigned int cols, double num);
Matrix::Matrix(unsigned int rows, unsigned int cols, double num):rows(rows),cols(cols){
	if (rows == 0 || cols == 0){
		printf("Error : Matrix size is wrong! : declare");
		return;
	}
	// 1. Allocate row array first
	this->at =(double**)malloc(sizeof(double*) * rows); 
	
	// 2. Then, allocate column 
	for (int i = 0; i < rows; i++)
		at[i] = (double*)malloc(sizeof(double) * cols);
	
	for(int row = 0; row<rows; row++){
		for(int col = 0; col<cols; col++){
			at[row][col] = num;
		}
	}
}

/*================================*/
/*            operand             */
/*================================*/

Matrix Matrix::transpose(){
	Matrix out(cols, rows);
	for(int row = 0; row< rows; row++){
		for(int col = 0; col<cols; col++){
			out.at[col][row] = at[row][col];
		}
	}
	return out;
}


void Matrix::swapRows(unsigned int row1, unsigned int row2){
	double cols = this->cols;
	double rows = this->rows;
	double tmp=0;
	for (int i=0; i<cols; i++){
		tmp = at[row1][i];
		at[row1][i] = at[row2][i];
		at[row2][i] = tmp;
	}
	return;
}

Matrix Matrix::operator + (Matrix mat) const{
	Matrix out = zeros(rows, cols);
	if (rows != mat.rows || cols != mat.cols) {
		printf("\n  ERROR!!: dimension error at 'add +' function\n");
		return Matrix(1, 1);
	}
	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
			out.at[i][j] = at[i][j] + mat.at[i][j];
	return out;
}


Matrix Matrix::operator - (Matrix mat) const{
	Matrix out = zeros(rows, cols);
	if (rows != mat.rows || cols != mat.cols) {
		printf("\n  ERROR!!: dimension error at 'sub -' function\n");
		return Matrix(1, 1);
	}
	for (int i = 0; i < out.rows; i++)
		for (int j = 0; j < out.cols; j++)
			out.at[i][j] = at[i][j] - mat.at[i][j];
	return out;
}

Matrix Matrix::operator * (Matrix mat)const{
	Matrix out = zeros(rows, mat.cols);

	if(cols != mat.rows){
		printf("\n ERROR : matrix size is wrong : operator * \n");
		return Matrix(0,0);
	}
	for(int row = 0; row<rows; row++){
		for(int col = 0; col<mat.cols; col++){
			for(int i=0; i<cols;i++){
				out.at[row][col] += this->at[row][i] * mat.at[i][col];
			}
		}
	}
	return out;
}



/*===============================================================*/
/*          				 function  				             */
/*===============================================================*/


/*================================*/
/*           file	              */
/*================================*/

//file directory
std::string GetHomeDirectory(){
    const char *homedir = std::getenv("HOME");
    if (homedir != nullptr) {
        return std::string(homedir);
	}else{
		std::cerr << "Error: HOME environment variable not set." << std::endl;
		return std::string(" ");
	}
    
}

// Free a memory allocated matrix
void freeMat(Matrix _A){
	// 1. Free allocated column memory
	for (int i = 0; i < _A.rows; i++)
		free(_A.at[i]);
	// 2. Free allocated row memory
	free(_A.at);
}

// Create a matrix from a text file
extern Matrix txt2Mat(std::string _filePath, std::string _fileName){
	std::ifstream file;
	std::string temp_string;
    std::string objFile = _filePath + _fileName + ".txt";
	int temp_int = 0, nRows = 0;
    std::cout<<objFile<<std::endl;

	file.open(objFile);
	if (!file.is_open()) {
		perror("??");
		printf("\n*********************************************");
		printf("\n  Could not access file: 'txt2Mat' function");
		printf("\n*********************************************\n");
		return Matrix();
	}
	while (getline(file, temp_string, '\t'))
		temp_int++;
	file.close();

	file.open(objFile);
	while (getline(file, temp_string, '\n'))
		nRows++;
	file.close();

	int nCols = (temp_int - 1) / nRows + 1;
	Matrix Out = zeros(nRows, nCols);

	file.open(objFile);
	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++) {
			file >> temp_string;
			Out.at[i][j] = stof(temp_string);
		}
	file.close();

	return Out;
}

/*================================*/
/*           declare              */
/*================================*/

// Create indentity matrix
extern Matrix eyes(unsigned int _row, unsigned int _col){
	Matrix out(_row, _col);
	for(int row = 0; row<out.rows; row++){
		for(int col = 0; col <out.cols; col++){
			if(row == col){
				out.at[row][col] = 1.;
			}else{
				out.at[row][col] = 0.;
			}
		}
	}
	return out;
}

// Create matrix of all zeros
extern Matrix zeros(unsigned int _row, unsigned int _col){
	Matrix out(_row, _col);
	for(int row = 0; row<out.rows; row++){
		for(int col = 0; col <out.cols; col++){
			out.at[row][col] = 0.;
		}
	}
	return out;
}

//intialize with all 1
extern Matrix ones(unsigned int _row, unsigned int _col){
	Matrix out(_row, _col);
	for(int i=0; i<out.rows;i++){
        for(int j=0;j<out.cols;j++){
            out.at[i][j]= 1.;
        }
    }
	return out;
}

// intialize with all _val
extern void initMat(Matrix& mat, double num){
	for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            mat.at[i][j] = num;
        }
    }
	return;
}


// convert array to Matrix
// double z0[2] = { 2.5, 2 };
// Z = arr2Mat(z0, n, 1);
Matrix	arr2Mat(double* _1Darray, int _rows, int _cols){
	// length(_1Darray) = _rows * _cols
	Matrix Output = Matrix(_rows, _cols);
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			Output.at[i][j] = _1Darray[i * _cols + j];
	return Output;
}

/*================================*/
/*             copy               */
/*================================*/

// copyMat matrix
extern void copyMat(Matrix& _A, Matrix& _B){
	Matrix Out = _A;
	_B = Out;
	return;
}

extern void copyVal(Matrix& _A, Matrix& _B){
	if(_A.rows> _B.rows || _A.cols > _B.cols){
		printf("ERROR : matrix 2 must be bigger than matrix 1 : copyVal\n");
	}
	for(int i=0; i<_A.rows;i++){
        for(int j=0;j<_A.cols;j++){
			_B.at[i][j]=_A.at[i][j];
        }
    }return;
}

/*================================*/
/*            operand             */
/*================================*/

//calculate norm
extern double norm(Matrix &A, int order = 2){
	int row = A.rows;
	int col = A.cols;
	if (col != 1 || order <1){
		printf("matrix size is wrong!!!!! can't calculate norm\n\n");
		return -1;
	}
	double norm_value = 0.0;
    for (int i = 0; i < row; i++) {
		for(int j=0; j<col; j++){
			norm_value += std::pow(std::abs(A.at[i][j]), order);
		}
    }
    return pow(norm_value, 1./order);
}

//calculate inverse matrix and check it
extern double invMat(Matrix& A, Matrix& Ainv){
	int n = A.rows;
    if (n != A.cols) {
        printf("its not squre matrix : invMat\n\n");
        return -1;
    }
	
	Ainv = zeros(n,n);
    Matrix L, U, P, y, x;

    LUdecomp(A, L, U, P);

    for (int i = 0; i < n; i++) {
		// i-th column of the identity matrix
		Matrix e = zeros(n,1);
		e.at[i][0] = 1.0;
		Matrix tmp = P*e;

		// Solve Ly = P*e for y using forward substitution
		fwdsub(L, tmp, y);

		//divide 0 trouble shooting
		try{
			// Solve Ux = y for x using backward substitution
			backsub(U, y, x);
		}
		catch(const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			printf("Error!!!: invmat \n\n");
			return -1;
		}
		
		// Set the i-th column of A_inv to the solution x
		for (int j = 0; j<n; j++){
			Ainv.at[j][i] = x.at[j][0];
		}
    }
	Matrix tmp = A*Ainv;
	bool flag = false;
	for(int i=0;i<tmp.rows;i++){
		if(tmp.at[i][i]==1){
			continue;
		}
		for(int j=0;j<tmp.cols;j++){
			if(tmp.at[i][j] != 0){
				flag =true;
			}
		}
	}
	if(flag){
		// printf("it's not inverse matrix\n");
		return -1;
	}else{
		// printf("it's inverse matrix\n");
		return 1;
	}
}

// Apply back-substitution
// Ux = y
extern void backsub(Matrix& U, Matrix &y, Matrix &x) {
    int n = U.rows;
	x = zeros(n,1);

    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.;
        for (int j = i + 1; j < n; j++) {
            sum += U.at[i][j] * x.at[j][0];
        }
		if (std::abs(U.at[i][i]) < 1e-10) {
            throw std::runtime_error("ERROR!!! Divided by zero : backsub\n\n");
        }
        x.at[i][0] = (y.at[i][0] - sum) / U.at[i][i]; 
    }
}

// Ly = b
extern void fwdsub(Matrix& L, Matrix &b, Matrix &y) {
    int n = L.rows;
    y = zeros(n, 1);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L.at[i][j] * y.at[j][0];
        }
		
        // Check for numerical instability (division by zero)
        if (std::abs(L.at[i][i]) < 1e-10){
            throw std::runtime_error("ERROR!!! Divided by zero : foward sub\n\n");
        }
		y.at[i][0] = (b.at[i][0] - sum) / L.at[i][i];
    }
}



/*================================*/
/*            gausselim           */
/*================================*/

// gaussElimination Ax = b -> Ux = d // return solution x
extern Matrix gaussElim(Matrix& A, Matrix& b, Matrix& U, Matrix& d){
	U = Matrix(A.rows, A.cols);
	copyVal(A,U);
	d = Matrix(b.rows, b.cols);
	copyVal(b,d);
	int n = A.rows;
	int k = A.cols;
	int a1 = U.rows;
	int a2 = U.cols;
	
	if((n != k) || (a1 != a2) ||  (a1 != n) || (n != b.rows)){
		printf("ERROR!!! matrix size is wrong : gaussElim\n\n");
		return Matrix();
	}
	for (int i = 0; i < n - 1; i++) {
        double pivot = U.at[i][i];
        if (pivot == 0.0) {
            printf("ERROR!!! Pivot is zero. Cannot continue : gaussElim \n\n");
            return Matrix();
        }
        for (int j = i + 1; j < n; j++) {
            double a = U.at[j][i] / pivot;
            for (int k = i; k < n; k++) {
                U.at[j][k] -= a * U.at[i][k];
            }
            d.at[j][0] -= a * d.at[i][0];
        }
    }
	Matrix x = Matrix(b.rows, b.cols);
	if(isUpperTriangle(U) == 0){
		printf("ERROR!!! matrix is not upper triangle! : gaussElimp \n\n");
		return Matrix();
	}
	backsub(U, d, x);
	return x;
}

// gaussElimination with pivoting pAx = b -> ux = d // return solution x
extern Matrix gaussElimp(Matrix& A, Matrix& b, Matrix& U, Matrix& d, Matrix& P){
	int n = A.rows;
	
	if((A.rows != A.cols) || (A.rows != b.rows)){
		printf("ERROR!!! matrix size is wrong : gaussElimp \n\n");
		return Matrix();
	}

	U = Matrix(A.rows, A.cols);
	copyVal(A,U);
	d = Matrix(b.rows, b.cols);
	copyVal(b,d);
	P = eyes(A.rows, A.cols);

	// p = identity matrix
	for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
			if(i==j){
				P.at[i][j] = 1;
			}else{
				P.at[i][j] = 0;
			}
        }
    }
	for (int i = 0; i < n; i++) {
		int maxRow = i;
        for(int j=i+1;j<n;j++){
			if(abs(U.at[j][i]) > abs(U.at[maxRow][i])){
				maxRow = j;
			}
		}
		U.swapRows(i,maxRow);
		d.swapRows(i,maxRow);
		P.swapRows(i,maxRow);
		
		for (int j = i + 1; j < n; j++) {
            double a = U.at[j][i] / U.at[i][i];
            for (int k = i; k < n; k++) {
                U.at[j][k] -= a * U.at[i][k];
            }
			d.at[j][0] -= a * d.at[i][0];
        }
	}
	Matrix x = Matrix(b.rows, b.cols);
	if(isUpperTriangle(U) == 0){
		printf("ERROR!!! matrix is not upper triangle! : gaussElimp \n\n");
		return Matrix();
	}
	backsub(U,d, x);
	return x;
}

//upperTriangle
int isUpperTriangle(Matrix& A) {
	if(A.rows != A.cols){
		return 0;  // Not square Matrix
	}
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < i; ++j) {
            if (A.at[i][j] != 0.0) {
                return 0;  // Not upper triangle
            }
        }
    }
    return 1;  // Upper triangle
}


//LUdecomp
extern void LUdecomp(Matrix &A, Matrix &L, Matrix &U, Matrix &P) {
    int n = A.rows;
    L = eyes(n, n); 
	U = Matrix(A.rows, A.cols);
	copyVal(A,U);
    P = eyes(n,n); 

    for (int k = 0; k < n - 1; k++) {

        int pivot_row = k;
		double max_val = std::abs(U.at[k][k]);
        //pivoting
        for (int i = k + 1; i < n; i++) {
			double current_val = std::abs(U.at[i][k]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = i;
            }
        }
        // swaprow in U and P
        if (pivot_row != k) {
            U.swapRows(k, pivot_row);
            P.swapRows(k, pivot_row);
			if(k>=1){
				// swaprow in L if k>=1
				L.swapRows(k, pivot_row);
			}
        }

        for (int i = k + 1; i < n; i++) {
            L.at[i][k] = U.at[i][k] / U.at[k][k];
            for (int j = k; j < n; j++) {
                U.at[i][j] -= L.at[i][k] * U.at[k][j];
            }
        }
    }
}

//pLUx = b
extern void solveLU(Matrix &L, Matrix &U, Matrix &P, Matrix &b, Matrix &x){
	int n = L.rows;
	Matrix y = zeros(n,1);
	for(int i=0; i<n;i++){
		for(int j = 0; j<n; j++){
			if (P.at[i][j] == 1){
				y.at[j][0] = b.at[i][0];
			}
		}
	}
	// Ly = Pb
    fwdsub(L, b, y);
    // Ux = y
	backsub(U, y, x);
}

// QRdecomb, Q(직교, 크기 1), R(나머지)
// Q=  [열백터 1,2,3,4,...]
// A = [열백터 1,2,3,4,...] -> 앞에것을 뒤에서 빼줌으로 직교화를 진행(벡터를 수직성분끼리 분해)
// R = [a1q1 a1q2 ...]
//	   [0    a2q2 ...]
extern void QRHousehold(Matrix& A, Matrix& Q, Matrix& R){
	//H = I - (2/(vt*v)) v*vt => I - (I/내적) * 외적
	// vt*v -> 내적 = (norm(v))**2
	// v = c + || c || e
	// c는 열백터 값, e는 r(1,1)부호값
	if(A.rows != A.cols || A.rows != Q.rows || A.cols != Q.cols || A.rows != R.rows || A.cols != R.cols){
		printf("Marix size is Error!! \n");
		return;
	}
    int n = A.rows;
    Q = eyes(n,n);
	R = Matrix(A.rows, A.cols);
	copyVal(A,R);

	Matrix c = zeros(A.rows, 1);
	double norm_c = 0.;
    // Initialize Q and R
    for (int j = 0; j < n-1; ++j) {
        Matrix c = zeros(n,1);
		Matrix e = zeros(n,1);

        // Copy column j of R to c
        for (int i = j; i < n; ++i) {
            c.at[i][0] = R.at[i][j];
        }

        // Set elements of e
        e.at[j][0] = (c.at[j][0] >= 0) ? 1.0 : -1.0;

		double norm_c = norm(c, 2);
		
        // Calculate v
        Matrix v = zeros(n,1);
		double vtv = 0.0;
        for (int i = 0; i < n; ++i) {
            v.at[i][0] = c.at[i][0] + e.at[i][0] * norm_c;
			vtv += v.at[i][0] * v.at[i][0];
        }
        // Calculate Householder transformation matrix H
        Matrix H = eyes(n,n);
		if (vtv==0.0){
			printf("ERROR!! Divided by 0 : QRHousehold \n");
			return;
		}
		double factor = 2.0 / vtv;
        for (int i = j; i < n; ++i) {
            for (int k = j; k < n; ++k) {
                H.at[i][k] -= factor * (v.at[i][0] * v.at[k][0]);
            }
        }
		// Update Q and R
        Q = Q*H;
        R = H*R;
	}
}

extern Matrix eig(Matrix &A){
	if( A.rows != A.cols){
		printf("Input Matrix is not square\n");
		return Matrix();
	}

	//tip QR분해를 한후 대각 행렬의 값들을 얻으면 끝!
	// det(A-ramda * I) = 0 => eigvalue
	Matrix A_tmp = A;
	Matrix Q = eyes(A_tmp.rows, A_tmp.cols);
	Matrix R = A_tmp;
	Matrix U = A_tmp;
	int N = 1000;
	for(int i=0;i<N;i++){
		QRHousehold(U,Q,R);
		U=R*Q;
		if(isUpperTriangle(U)){
			break;
		}
	}
	Matrix Out = zeros(U.cols,1);
	for(int i=0;i<R.cols;i++){
		Out.at[i][0] = U.at[i][i];
	}
	return Out;
}

extern Matrix eigvec(Matrix& A){
	if(A.rows != A.cols){
		printf("matrix is not square!!\n");
		return Matrix();
	}
	if(!(A.rows>1 && A.rows<4)){
		printf("matrix size must be 2 or 3\n");
		return Matrix();
	}
	Matrix eigis = eig(A);
	double eigvalue_iter = eigis.rows;
	double eigvalue;

	Matrix Out = zeros(A.rows, A.cols);
	Matrix b = zeros(A.rows, A.cols);
	for(int iter = 0; iter<eigvalue_iter; iter++){
		eigvalue = eigis.at[iter][0];
		//A-ramda*I
		copyVal(A,b);
		for(int i = 0; i<A.rows; i++){
			b.at[i][i] = A.at[i][i]-eigvalue;
		}
		
		//initialize v
		Matrix v = ones(A.rows,1);
		Matrix v_tmp = ones(A.rows-1,1);
		//u*v_tmp = d
		Matrix d = ones(A.rows-1,1);
		Matrix u = ones(A.rows-1,A.rows-1);
		
		int tmpd=0;
		int tmpu1=0;
		int tmpu2=0;
		
		//make u, d
		for(int j=0; j<A.rows;j++){
			if(j==0){
				continue;
			}
			for(int k=0; k<A.cols;k++){
				if(k==0){
					d.at[tmpd][0] = -b.at[j][k];
					tmpd+=1;
					continue;
				}
				u.at[tmpu1][tmpu2] = b.at[j][k];
				tmpu2+=1;
				if(tmpu2 == u.cols){
					tmpu1+=1;
					tmpu2=0;
				}
			}
		}
		// v = inv(u)*d
		// inv함수로 대체 할 수 있는지 확인 필요
		if(eigvalue_iter == 2){
			double det = u.at[0][0];
			if(det == 0){
				printf("Matrix is not independent : Error in Eigvec!!\n");
				return Matrix();
			}
			v_tmp.at[0][0] = d.at[0][0]/det;
		}else if (eigvalue_iter == 3){
			double tmp = u.at[0][0];
			u.at[0][0] = u.at[1][1];
			u.at[1][1] = tmp;
			u.at[0][1] = -u.at[0][1];
			u.at[1][0] = -u.at[1][0];
			double det = u.at[0][0]*u.at[1][1]-u.at[1][0]*u.at[0][1];
			if(det ==0){
				printf("Matrix is not independent : Error in Eigvec!\n");
				return Matrix();
			}
			v_tmp.at[0][0] = (u.at[0][0]*d.at[0][0]+u.at[1][0]*d.at[1][0])/det;
			v_tmp.at[1][0] = (u.at[1][0]*d.at[0][0]+u.at[1][1]*d.at[1][0])/det;
		}
		//make v
		int tmp_index=0;
		for(int j=0; j<A.rows;j++){
			if(j!=0){
				v.at[j][0] = v_tmp.at[tmp_index][0];
				tmp_index++;
			}
		}
		double normv = norm(v,2);
		for(int tmp=0; tmp<A.rows;tmp++){	
			Out.at[tmp][iter] = v.at[tmp][0]/normv;
		}
	}
	return Out;
}

//LinearFit
extern void linearFit(Matrix& Z, Matrix& X, Matrix& Y, int order){
	//X, Y = nx1 matrix
	if(X.rows == 1){
		X = X.transpose();
	}
	if(Y.rows == 1){
		Y = Y.transpose();
	}
	int m = X.rows;
	int n = order;
	
	if( X.rows != Y.rows){
		printf("Matrix size is not equal X,Y : LinearFit\n\n");
		return;
	}
	if ((1>order || order > m)){
		printf("can't solve the problem order is wrong : LinearFit\n\n");
		return;
	}
	//a0 ~ an
	Z = Matrix(n+1, 1);
	double sumx[2*n+1];
	for(int i =0; i<n+1; i++){
		for (int j = 0; j<m; j++){
			sumx[i] += pow(X.at[j][0], i);
			sumx[i+order+1] += pow(X.at[j][0], i+order+1);
		}
	}
	Matrix ATA =  Matrix(n+1, n+1);
	Matrix ATY =  Matrix(n+1, 1);

	for(int i=0; i<n+1; i++){
		for(int j=0; j<n+1; j++){
			ATA.at[i][j] = sumx[i+j];
		}
		for(int j=0; j<m; j++){
			ATY.at[i][0] += pow(X.at[j][0], i) * Y.at[j][0];
		}
	}
	Matrix U, d, p;
	Z = gaussElimp(ATA, ATY, U, d, p);
	
	return;
}

extern double callinearFit(Matrix& Z, double x){
	// Z = nx1 Matrix
	if (Z.rows == 1 || Z.cols == 1){
		if(Z.rows == 1){
			Z = Z.transpose();
		}
		int n = Z.rows;
		double output=0;
		for (int i=0; i<n; i++){
		output += Z.at[i][0] * pow(x,i);
		}
		return output;
	}
	printf("wrong Marix input!! : calculatelinearFit \n\n");
	return -1;
}


extern void expFit(Matrix& Z, Matrix& X, Matrix& Y, int order){
	//X, Y = nx1 matrix
	if(X.rows == 1){
		X = X.transpose();
	}
	if(Y.rows == 1){
		Y = Y.transpose();
	}
	int m = Y.rows;
	
	if( X.rows != Y.rows){
		printf("Matrix size is not equal X,Y : LinearFit\n\n");
		return;
	}
	if ((1>order || order > m)){
		printf("can't solve the problem order is wrong : LinearFit\n\n");
		return;
	}
	Matrix y_tmp =Matrix(Y.rows, Y.cols);
	for(int i=0; i<m; i++){
		y_tmp.at[i][0] = log(Y.at[i][0]);
	}
	
	linearFit(Z,X,y_tmp,order);
	// Z.print("Z");
	Z.at[0][0] = exp(Z.at[0][0]);
	// Z.print("Z");
	// printf("C0 = %f, \t\t C1 = %f\n", Z.at[0][0], Z.at[1][0]);
	
}

Matrix jacob(Matrix f(Matrix _x), Matrix x){
	if(f(x).cols != 1 || x.cols != 1 || f(x).rows != x.rows){
		printf("ERROR : ~~ \n\n");

		return Matrix(0,0);
	}
	int size = x.rows;
	double h = 1e-8;
	Matrix jacob_out = Matrix(size, size);
	Matrix tmp_x1;
	Matrix tmp_x2;

	for(int row = 0; row<size; row++){
		// tmp_x1 = copyMat(x);
		// tmp_x2 = copyMat(x);
		tmp_x1.vec(row) +=h;
		tmp_x2.vec(row) -=h;
		for (int col=0; col<size; col++){
			jacob_out.at[row][col] = (f(tmp_x1).vec(col) - f(tmp_x2).vec(col)) / (2.*h);
		}
	}
	return jacob_out;
}
