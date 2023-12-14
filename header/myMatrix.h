/*----------------------------------------------------------------\
@ Numerical Programming by Young-Keun Kim - Handong Global University

Author           : Lee Chan Keun
Created          : 26-03-2018
Modified         : 14-12-2023
Language/ver     : C++ in MSVS2019

Description      : myMatrix.h
----------------------------------------------------------------*/

#ifndef		_MY_MATRIX_H		// use either (#pragma once) or  (#ifndef ...#endif)
#define		_MY_MATRIX_H

#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
struct Matrix{ 
public:
	//2D array
	double** at;
	int rows, cols;

	void print(char* name);

	/*================================*/
	/*           declare              */
	/*================================*/

	Matrix();
	Matrix(unsigned int rows, unsigned int cols);
	Matrix(unsigned int rows, unsigned int cols, double num);

	/*================================*/
	/*            operand             */
	/*================================*/

	Matrix transpose();
	//swaprows row1 with row2
	void swapRows(unsigned int row1, unsigned int row2);
	Matrix operator + (Matrix mat) const;
	Matrix operator - (Matrix mat) const;
	Matrix operator * (Matrix mat) const;
};

/*================================*/
/*           file	              */
/*================================*/

//file directory
extern std::string GetHomeDirectory();

// Free a memory allocated matrix
extern void freeMat(Matrix _A);

// Create a matrix from a text file
extern	Matrix	txt2Mat(std::string _filePath, std::string _fileName);

/*================================*/
/*           declare              */
/*================================*/

// Create indentity matrix
extern Matrix eyes(unsigned int rows, unsigned int cols);

// Create matrix of all zeros
extern Matrix zeros(unsigned int rows, unsigned int cols);

// Create matrix of all ones
extern Matrix ones(unsigned int _rows, unsigned int _cols);

// initialization of Matrix elements
extern void initMat(Matrix& mat, double num);

// convert array to Matrix
Matrix	arr2Mat(double* _1Darray, int _rows, int _cols);

/*================================*/
/*             copy               */
/*================================*/

// Copy matrix
extern void copyMat(Matrix& _A, Matrix& _B);

// Copy matrix Elements from A to B
extern void copyVal(const Matrix& _A, Matrix& _B);


/*================================*/
/*            operand             */
/*================================*/

//calculate norm
extern double norm(Matrix &A, int order);

//calculate inverse matrix and check it
extern double invMat(Matrix& A, Matrix& Ainv);

// back-substitution
extern void backsub(Matrix& U, Matrix &y, Matrix &x);

// forward-substitution
extern void fwdsub(Matrix& L, Matrix &b, Matrix &y);


/*================================*/
/*            gausselim           */
/*================================*/



// gaussElimination Ax = b -> Ux = d // return solution x
extern Matrix gaussElim(Matrix& A, Matrix& b, Matrix& U, Matrix& d);

// gaussElimination with pivoting pAx = b -> ux = d // return solution x
extern Matrix gaussElimp(Matrix& A, Matrix& b, Matrix& U, Matrix& d, Matrix& P);

extern int isUpperTriangle(Matrix& A);

//LUdecomp
void LUdecomp(Matrix &A, Matrix &L, Matrix &U, Matrix &P);

//solveLU
extern void solveLU(Matrix &L, Matrix &U, Matrix &P, Matrix &b, Matrix &x);

//QRdecomp
extern void QRHousehold(Matrix& A, Matrix& Q, Matrix& R);


//eigvalue && eigvector
extern Matrix eig(Matrix &A);
extern Matrix eigvec(Matrix& A);

//LinearFit
extern void linearFit(Matrix& Z, Matrix& X, Matrix& Y, int order);
extern double callinearFit(Matrix& Z, double x);
extern void expFit(Matrix& Z, Matrix& X, Matrix& Y, int order);


#endif