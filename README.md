# API : Documentation

# **Numerical Programming API**

```cpp
#include "myNP.hpp"
#include "myMatrix.h"
```

# myNP.hpp

## Operand

### **rad2deg()**

change radian to degree

```cpp
double rad2deg(double _x);
```

**Parameters**

- _x : initial value

**example Code**

```cpp
double rad = PI;
double degree = 0;

degree = rad2deg(rad);
```

### **deg2rad()**

change degree to radian

```cpp
double deg2rad(double _x);
```

**Parameters**

- _x : initial value

**example Code**

```cpp
double rad = 0;
double degree = 180;

rad = deg2rad(degree);
```

### power()

calculate power function

```cpp
double power(double _x, int N);
```

**Parameters**

- _x : initial value
- N : iteration → N must be positive integer

**example Code**

```cpp
double x = 2;
int N = 10;
double output=0;

// output = 1024
output = power(2,10);
```

### factorial()

calculate factorial

```cpp
extern double factorial(int N);
```

**Parameters**

- N : initial value → must be positive integer or 0

**example Code**

```cpp
int N = 3;
double output=0;

// output = 6
output = factorial(N);
```

### printVec()

print Vector

```cpp
void printVec(double* _vec, int _row);
```

**Parameters**

- _vec : array → length of array must be more than 1
- _row : integer value → row must have same value of length of array

**example Code**

```cpp
#define N int(12)
double y[N] = { -3.632, -0.3935, 1, 0.6487, -1.282, -4.518, -8.611, -12.82, -15.91, -15.88, -9.402, 9.017};

printVec(y, N);
```

### sinTaylor()

Taylor series approximation for sin(x) using pre-defined functions (input unit: [rad])

```cpp
double sinTaylor(double _x);
```

**Parameters**

- _x : initial value ([rad])

**example Code**

```cpp
double rad = PI;
double out = 0;

out = sinTaylor(rad);
```

### sindTaylor()

Taylor series approximation for sin(x) using pre-defined functions (input unit: [deg])

```cpp
double sindTaylor(double _x);
```

**Parameters**

- _x : initial value ([deg])

**example Code**

```cpp
double deg = 180;
double out = 0;

out = sindTaylor(deg);
```

### cosTaylor()

Taylor series approximation for cos(x) using pre-defined functions (input unit: [rad])

```cpp
double cosTaylor(double _x);
```

**Parameters**

- _x : initial value ([rad])

**example Code**

```cpp
double rad = PI;
double out = 0;

out = cosTaylor(rad);
```

### cosdTaylor()

Taylor series approximation for cos(x) using pre-defined functions (input unit: [deg])

```cpp
double cosdTaylor(double _x);
```

**Parameters**

- _x : initial value ([deg])

**example Code**

```cpp
double deg = 180;
double out = 0;

out = sindTaylor(deg);
```

## Non-linear solver

### bisection()

Solves the non-linear problem using bisection method

```cpp
extern double bisection(double func(double),float _a0, float _b0, float _tol);
```

**Parameters**

- func(double) : initial function
- _a0 : starting point
- _b0 : ending point

+) must func(_a0)*func(_b0)<0 

- _tol : initialize tolerance

**example Code**

```cpp
double func(double theta);

double a = 0;
double b = 1;
double sol_b;
double tol = 0.00001;

sol_b = bisection(func, a, b, tol);
```

### newtonRaphson()

Solves the non-linear problem using newtonRaphson method

```cpp
extern double newtonRaphson(double func(double), double dfunc(double), double _x0, double _tol);
```

**Parameters**

- func(double) : initial function
- dfunc(double) : initial differential function
- _x0 : starting point
- _tol : initialize tolerance

**example Code**

```cpp
double func(double x);
double dfunc(double x);

double x0 = 0;
double sol_nr;
double tol = 0.00001;

sol_nr = newtonRaphson(func,dfunc,x0, tol);
```

### secant()

Solves the non-linear problem using secant method

```cpp
extern double secant(double func(double), double _x0, double _x1, double _tol);
```

**Parameters**

- func(double) : initial function
- _x0 : starting point(range)
- _x1 : ending point(range)
- _tol : initialize tolerance

**example Code**

```cpp
double func(double x);

double x0 = 0;
double x1 = 1;
double sol_nr;
double tol = 0.00001;

sol_nr = secant(func,x0, x1, tol);
```

## Distribution

### gradient1D()

Solves distribution in 1 dimension

```cpp
extern void gradient1D(double x[], double y[], double dydx[], int m);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- dydx[] : output distributed value
- m : length of array(length of x[], y[] is same)

**Example Code**

```cpp
double X[12]=[-1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5];
double Y[12]=[-3.632 -0.3935 1 0.6487 -1.282 -4.518 -8.611 -12.82 -15.91 -15.88 -9.402 9.017];

double dydt[12] = {0};

gradient1D(X, Y, dydt, 12);
printVec(dydt, 12);
```

### acceleration()

Solves second distribution in 1 dimension

```cpp
extern void acceleration(double x[ ], double y[ ], double dydx[ ], int m);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- dydx[] : output second distributed value
- m : length of array(length of x[], y[] is same)

**Example Code**

```cpp
double X[12]=[-1 -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5];
double Y[12]=[-3.632 -0.3935 1 0.6487 -1.282 -4.518 -8.611 -12.82 -15.91 -15.88 -9.402 9.017];

double dydt[12] = {0};

acceleration(X, Y, dydt, 12);
printVec(dydt, 12);
```

### gradientFunc()

Solves distribution in 1 dimension with function

```cpp
// Truncation error should be O(h^2) 
extern void gradientFunc(double func(const double x), double x[], double dydx[], int m);
```

**Parameters**

- func(double) : initial function
- x[] : input x_points
- dydx[] : output second distributed value
- m : length of array(length of x[], dydx[] is same)

**Example Code**

```cpp
double myFunc(const double x);

double t[21] = {0};
	for (int i = 0; i < 21; i++) t[i] = 0.2 * i;
double dydt[21] ={ 0 };

gradientFunc(myFunc, t ,dydt, 21);
```

## Integration

### trapz()

Solves integration in simple method

```cpp
extern double trapz(double x[], double y[], int m);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- m : length of array(length of x[], y[] is same)

**Example Code**

```cpp
double x[] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 };
double y[] = { 0, 3, 8, 20, 33, 42, 40, 48, 60, 12, 8, 4, 3 };

int M = sizeof(x) / sizeof(x[0]);
double I_trapz = 0;
I_trapz=trapz(x, y, M);
printf("I_trapz = %f\n\n", I_trapz);
```

### simpson13()

Solves integration in simpson13 method (m is odd)

```cpp
extern double simpson13(double x[], double y[], int m);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- m : length of array(length of x[], y[] is same)

m ≥2 && m is odd number

**Example Code**

```cpp
double X[] = {-3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3};
double Y[] = {0, 2.1875, 3.75, 4.6875, 5, 4.6875, 3.75, 2.1875, 0};
double I_simpson = 0;
int N = sizeof(X) / sizeof(X[0]);

I_simpson = simpson13(X, Y, N);
printf("I_simpson13  = %f\n\n", I_simpson);
```

### simpson38()

Solves integration in simpson38 method(m is even)

```cpp
extern double simpson38(double x[], double y[], int m);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- m : length of array(length of x[], y[] is same)

m ≥3 && m is even number

**Example Code**

```cpp
double X[] = {-3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3, 3.75};
double Y[] = {0, 2.1875, 3.75, 4.6875, 5, 4.6875, 3.75, 2.1875, 1.5, 0};
double I_simpson = 0;
int N = sizeof(X) / sizeof(X[0]);

I_simpson = simpson38(X, Y, N);
printf("I_simpson38  = %f\n\n", I_simpson);
```

### integral()

Solves integration in simpson13 method with function

```cpp
extern double integral(double func(const double), double a, double b, int n);
```

**Parameters**

- func(double) : initial function
- a : starting point
- b : ending point
- n : accuracy(The number of times divide the interval between a and b)

**Example Code**

```cpp
double myFunc(double x);
double I_function = 0;
double funcX[SIZE] = {0};
double funcY[SIZE] = {0};
for(int i=0; i<SIZE; i++){
	funcX[i] =  -1 + double(i) * 2/SIZE;
}

I_function=integral(myFunc, -1, 1, SIZE);
printf("I_function  = %f\n\n", I_function);
```

## Interpolation

Interpolation refers to estimating a value located between values of known points from known values.

### lagrageInterpolation()

Solves Interpolation with lagrange method

![https://github.com/lcg0070/NumP/blob/main/images/Lagrange.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/Lagrange.png?raw=true)

```cpp
extern double lagrageInterpolation(double* x, double* y, double xq, int m, int order);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- xq : Coordinate of the value, want to find
- m : length of array(length of x[], y[] is same)
- order : order of interpolation

**Example Code**

```cpp
double x[] = { 0, 10, 20, 30, 40, 50, 60, 70, 80};
double y[] = { 0.94, 0.96, 1, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21};

int M = sizeof(x) / sizeof(x[0]);
double xq = 25;
double value = 0;
value = lagrageInterpolation(x, y, xq, M, 1)
```

### newtonInterpolation()

Solves Interpolation with newton method

![https://github.com/lcg0070/NumP/blob/main/images/newtonPolynomial.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/newtonPolynomial.png?raw=true)

```cpp
double newtonInterpolation(double* x, double* y, double xq, int m, int order);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- xq : Coordinate of the value, want to find
- m : length of array(length of x[], y[] is same)
- order : order of interpolation

**Example Code**

```cpp
double x[] = { 0, 10, 20, 30, 40, 50, 60, 70, 80};
double y[] = { 0.94, 0.96, 1, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21};

int M = sizeof(x) / sizeof(x[0]);
double xq = 25;
double value = 0;
value = newtonInterpolation(x, y, xq, M, 1)
```

### linearSpline()

Solves Interpolation with linearspline

![https://github.com/lcg0070/NumP/blob/main/images/linearspline.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/linearspline.png?raw=true)

```cpp
extern double linearSpline(double* x, double* y, int n, double xq);
```

**Parameters**

- x[] : input x_points
- y[] : input y_points
- xq : Coordinate of the value, want to find
- n : length of array(length of x[], y[] is same)

**Example Code**

```cpp
double x[] = { 0, 10, 20, 30, 40, 50, 60, 70, 80};
double y[] = { 0.94, 0.96, 1, 1.05, 1.07, 1.09, 1.14, 1.17, 1.21};

int M = sizeof(x) / sizeof(x[0]);
double xq = 25;
double value = 0;
value = linearSpline(x, y, M, xq)
```

## ODE-IVP

solve Ordinary Differential Equations

### odeRK2()

Solves ODE with 2nd order Runge-Kutta method

```cpp
extern void odeRK2(double myfunc(const double, const double), double y[], double t0, double tf, double h, double y0);
```

**Parameters**

- myfunc(double, double) : initial function (parameter → t, y) = dy/dt
- y[] : output y_points
- t0 : starting point
- tf : ending point
- h : intervals
- y0 : initial value of coordinate

**Example Code**

```cpp
double myfunc(double x, double y);

double y[101];
double a= 0;
double b = 0.1;

double h = 0.001;

odeRK2(myfunc,y,a,b,h,0);
for(int i =0; i<101; i++){
	printf("%f\n", y[i]);
}
```

### ode()

Solves ODE with 3rd order Runge-Kutta method

```cpp
//3rd order Runge-Kutta method 
extern void ode(double myfunc(const double t, const double y), double y[], double t0, double tf, double h, double y0);
```

**Parameters**

- myfunc(double, double) : initial function (parameter → t, y) = dy/dt
- y[] : output y_points
- t0 : starting point
- tf : ending point
- h : intervals
- y0 : initial value of coordinate

**Example Code**

```cpp
double myfunc(double x, double y);

double y[101];
double a= 0;
double b = 0.1;

double h = 0.001;

ode(myfunc,y,a,b,h,0);
for(int i =0; i<101; i++){
	printf("%f\n", y[i]);
}
```

### sysRK2()

Solves ODE with 2nd order sys RK2

```cpp
extern void sysRK2(void myfunc(const double, double[], double[]), double y[], double z[], double a, double b, double h, double yINI, double zINI);
```

**Parameters**

- myfunc(double, double[], double[]) : initial function (parameter → x, y[], z[]) = dy/dt, dz/dt
- y[] : output y_points
- z[] : output z_points
- a : starting point
- b : ending point
- h : intervals
- yINI : initial value of y0
- zINI : initial value of z0

**Example Code**

```cpp
double z[101];
a = 0;
b = 1;
h = 0.01;
sysRK2(mckfunc, y,z,a,b,h,0,0.2);

for(int i =0; i<101; i++){
    printf("%f\n", y[i]);
}
printf("\n\n");
for(int i =0; i<101; i++){
    printf("%f\n", z[i]);
}
```

# myMatrix.h

## Method

### print**()**

print matrix

```cpp
//print Matrix
void Matrix::print(char* name);
```

**Parameters**

- name : initialize name

**example Code**

```cpp
Matrix a(1,1);

a.print("Marix : A")
```

### Declare Matrix

declare Matrix case by case

```cpp
// Matrix();
Matrix::Matrix();

// Matrix(unsigned int rows, unsigned int cols);
Matrix::Matrix(unsigned int rows, unsigned int cols):rows(rows),cols(cols);

// Matrix(unsigned int rows, unsigned int cols, double num);
Matrix::Matrix(unsigned int rows, unsigned int cols, double num):rows(rows),cols(cols);
```

**Parameters**

- rows : initialize row
- cols : initialize column
- num : input initial value

**example Code**

```cpp
Matrix a;
Matrix b(1,2);
Matrix c(3,3,4);
```

### Operand

useful operand of matrix

```cpp
Matrix Matrix::transpose();

void Matrix::swapRows(unsigned int row1, unsigned int row2);

Matrix Matrix::operator + (Matrix mat) const;

Matrix Matrix::operator - (Matrix mat) const;

Matrix Matrix::operator * (Matrix mat)const;
```

**Parameters**

swapRows

- row1, row2 : change value of row1 and row2

**example Code**

```cpp
Matrix a(3,1,2);
Matrix b(3,3,3);
Matrix c(3,3,1);

Matrix trans = a.transpose();
trans.swapRows(1,2);

Matrix sum_bc = b+c;
Matrix sub_bc = b-c;
Matrix mul_bc = b*c;
```

## File and Memory

### GetHomeDirectory**()**

Get Home Directory to read file

```cpp
//file directory
std::string GetHomeDirectory();
```

**Parameters**

- none

**example Code**

```cpp
string homedir = GetHomeDirectory();
```

### txt2Mat**()**

change txt file to Matrix form

```cpp
extern Matrix txt2Mat(std::string _filePath, std::string _fileName);
```

**Parameters**

- none

**example Code**

```cpp
Matrix A = txt2Mat(path, "prob_A");
```

### freeMat**()**

free memory of matrix

```cpp
void freeMat(Matrix _A);
```

**Parameters**

- none

**example Code**

```cpp
Matrix A(4,1,3);
freeMat(A);
```

## Declare Matrix

### eyes**()**

Make identity Matrix

```cpp
extern Matrix eyes(unsigned int _row, unsigned int _col);
```

**Parameters**

- _row : initialize row
- _cols : initialize column

**example Code**

```cpp
Matrix A = eyes(3,3);
```

### zeros**()**

Create matrix of all zeros

```cpp
extern Matrix zeros(unsigned int _row, unsigned int _col);
```

**Parameters**

- _row : initialize row
- _cols : initialize column

**example Code**

```cpp
Matrix A = zeors(3,3);
```

### ones**()**

Create matrix of all 1

```cpp
extern Matrix ones(unsigned int _row, unsigned int _col);
```

**Parameters**

- _row : initialize row
- _cols : initialize column

**example Code**

```cpp
Matrix A = ones(3,3);
```

### initMat**()**

fill matrix with num

```cpp
extern void initMat(Matrix& mat, double num);
```

**Parameters**

- mat : target matrix
- num : initialize number

**example Code**

```cpp
Matrix A = ones(3,3);
initMat(A, 2);
```

### arr2Mat**()**

convert array to Matrix

```cpp
Matrix	arr2Mat(double* _1Darray, int _rows, int _cols);
```

**Parameters**

- _1Darray : target array(must 1Darray, length of array must be _rows*_cols)
- _rows : initialize row
- _cols : initialize column

**example Code**

```cpp
double arr = {1,2,3,4,5,6}

Matrix A = arr2Mat(arr, 2,3);
```

## Copy Matrix

### copyMat**()**

copy Matrix  A to Matrix B

```cpp
extern void copyMat(Matrix& _A, Matrix& _B);
```

**Parameters**

- _A : initial Matrix
- _B : target Matrix

**example Code**

```cpp
Matrix A = zeros(3,3);
Matrix B;

copyMat(A, B);

```

### copyVal**()**

copy value of Matrix  A to Matrix B

```cpp
extern void copyVal(Matrix& _A, Matrix& _B);
```

**Parameters**

- _A : initial Matrix
- _B : target Matrix ( size of Matrix B must be bigger than A)

**example Code**

```cpp
Matrix A = zeros(3,3);
Matrix B = ones(5,4);

copyVal(A, B);
```

## Operand of Matrix

### norm**()**

normalize the matrix by order

```cpp
extern double norm(Matrix &A, int order = 2);
```

**Parameters**

- _A : initial Matrix
- order : initial order(formula will be different by order)

**example Code**

```cpp
Matrix A = eyes(3,3);
double nor = norm(A, 2);
```

### invMat**()**

find inverse Matrix by using LUdecomposition

```cpp
extern double invMat(Matrix& A, Matrix& Ainv);
```

**Parameters**

- A : initial Matrix
- Ainv : target Matrix

**example Code**

```cpp
Matrix A = eyes(3,3);
Matrix Ainv;

invMat(A, Ainv);
```

### backsub**()**

back substitution of Matrix(Matrix must be upper triangle matrix)

```cpp
extern void backsub(Matrix& U, Matrix &y, Matrix &x);
```

**Parameters**

- U : initial Matrix
- y : input vector
- x : output vector

Ux = y

**example Code**

```cpp
Matrix upper = zeros(3,3); //need to intialize value
Matrix vec(3,1); //need to intialize value
Matrix vecout(3,1)

backsub(upper, vecout, vec);
```

### fwdsub**()**

forward substitution of Matrix(Matrix must be lower triangle matrix)

```cpp
extern void fwdsub(Matrix& L, Matrix &b, Matrix &y);
```

**Parameters**

- L : initial Matrix
- b : input vector
- y : output vector

Ly = b

**example Code**

```cpp
Matrix low = zeros(3,3); //need to intialize value
Matrix vec(3,1); //need to intialize value
Matrix vecout(3,1)

fwdsub(low, vecout, vec);
```

## Gauss Elimination of Matrix

### gaussElim**()**

solve matrix using gauss elimination

```cpp
extern Matrix gaussElim(Matrix& A, Matrix& b, Matrix& U, Matrix& d);
```

**Parameters**

- A : initial Matrix
- b : initial vector
- U : output Matrix(upper triangle matrix)
- d : output vector

Ax = b → Ux = d

after back substitution we can get x(vector)

**example Code**

```cpp
Matrix A(3,3); //need to intialize value
Matrix b(3,1); //need to intialize value
Matrix U, d;
Matrix solve_mat;

solve_mat = gaussElim(A, b, U, d);
```

### gaussElimp**()**

solve matrix using gauss elimination + pivoting(if division is zero it changes rows)

```cpp
extern Matrix gaussElimp(Matrix& A, Matrix& b, Matrix& U, Matrix& d, Matrix& P);
```

**Parameters**

- A : initial Matrix
- b : initial vector
- U : output Matrix(upper triangle matrix)
- d : output vector
- p : output Matrix(pivot matrix)

pAx = b → Ux = d

after back substitution we can get x(vector)

**example Code**

```cpp
Matrix A(3,3); //need to intialize value
Matrix b(3,1); //need to intialize value
Matrix p, U, d;
Matrix solve_mat;

solve_mat = gaussElimp(A, b, U, d, p);
```

## Decomposition of Matrix

### LUdecomp**()**

make Matrix into Lower triangle Matrix and Upper triangle Matrix

```cpp
extern void LUdecomp(Matrix &A, Matrix &L, Matrix &U, Matrix &P);
```

**Parameters**

- A : initial Matrix
- L : output Matrix(lower triangle matrix)
- U : output Matrix(upper triangle matrix)
- p : output Matrix(pivot matrix)

pAx = LU

**example Code**

```cpp
Matrix A(3,3); //need to intialize value
Matrix p, L, U;

LUdecomp(A,L,U,p)
```

### solveLU**()**

solve Matrix if Lower triangle Matrix and Upper triangle Matrix is multiplied

```cpp
extern void solveLU(Matrix &L, Matrix &U, Matrix &P, Matrix &b, Matrix &x);
```

**Parameters**

- L : input Matrix(lower triangle matrix)
- U : input Matrix(upper triangle matrix)
- p : input Matrix(pivot matrix)
- b : input vector
- x : output vector

pLUx = b

**example Code**

```cpp

Matrix matK = txt2Mat(path, "prob1_matK");
Matrix vecf = txt2Mat(path, "prob1_vecf");
    
Matrix l;
Matrix u;
Matrix p;
Matrix x;
// Ax = pLUx
LUdecomp(matK,l,u,p);

// Ax = pLUx = b
solveLU(l,u,p,vecf,x);
```

### QRHousehold**()**

make Matrix into Q Matrix and R Matrix, (its about projection)

![https://github.com/lcg0070/NumP/blob/main/images/QRdecomb.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/QRdecomb.png?raw=true)

```cpp
extern void QRHousehold(Matrix& A, Matrix& Q, Matrix& R);
```

**Parameters**

- A : input Matrix
- Q : output Matrix(column vectors : normalized)
- R : output Matrix(upper triangle matrix)

A = QR

**example Code**

```cpp

Matrix A = txt2Mat(path, "prob1_matK");

Matrix Q, R; 

QRHousehold(A,Q,R);  
```

## Eigenvalue and Eigenvector

### eig**()**

find eigenvalue of Matrix

```cpp
extern Matrix eig(Matrix &A);
```

**Parameters**

- A : input Matrix

**example Code**

```cpp

Matrix A = txt2Mat(path, "prob1_matK");

Matrix eigvalue;

eigvalue = eig(A);
```

### eigvec**()**

find eigenvector of Matrix(Matrix size must be 2x2 or 3x3)

```cpp
extern Matrix eigvec(Matrix& A);
```

**Parameters**

- A : input Matrix

**example Code**

```cpp

Matrix A = txt2Mat(path, "prob1_matK");

Matrix eigvector;
eigvalue = eigvec(A);
```

## Linear Regression

### linearFit**()**

solve Linear regression by order, with input coordinates

![https://github.com/lcg0070/NumP/blob/main/images/linearfit.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/linearfit.png?raw=true)

```cpp
extern void linearFit(Matrix& Z, Matrix& X, Matrix& Y, int order);
```

**Parameters**

- Z : output Matrix ( a0~ an value)
- X : input Matrix (input coordinate)
- Y : input Matrix (input coordinate)
- order : initialize order (must be integer)

z = inv(aTa) * aTy

**example Code**

```cpp
double strain[16] = {};
for (int i=0; i<16; i++){
	strain[i] = 0.4*i;
}
double stress[16] = {0, 3, 4.5, 5.8, 5.9, 5.8, 6.2, 7.4, 9.6, 15.6, 20.7, 26.7, 31.1, 35.6, 39.3, 41.5};
Matrix X = arr2Mat(strain, 1, 16);
Matrix Y= arr2Mat(stress, 1, 16);

Matrix Z;
linearFit(Z,X,Y,4);
```

### callinearFit**()**

calculate the value of the formula, in coordinate x

```cpp
extern double callinearFit(Matrix& Z, double x);
```

**Parameters**

- Z : input (linear Regression Matrix)
- x : input coordinate

output = z[0] * x^n + z[1] * x^(n-1) …

**example Code**

```cpp
double T[6]= {30, 40, 50, 60, 70, 80}; 
double P[6]= {1.05, 1.07, 1.09, 1.14, 1.17, 1.21}; 

Matrix X1 = arr2Mat(T, 1, 6);
Matrix Y1 = arr2Mat(P, 1, 6);

Matrix Z1;
linearFit(Z1,X1,Y1,1);
printf("%f\n", callinearFit(Z1,100));
```

### expFit**()**

solve Linear regression in order 2, exponential function

![https://github.com/lcg0070/NumP/blob/main/images/expfit.png?raw=true](https://github.com/lcg0070/NumP/blob/main/images/expfit.png?raw=true)

```cpp
extern void expFit(Matrix& Z, Matrix& X, Matrix& Y, int order=2);
```

**Parameters**

- Z : output Matrix ( a0~ an value)
- X : input Matrix (input coordinate)
- Y : input Matrix (input coordinate)
- order : initialize order (must be integer)

output = z[0] * x^n + z[1] * x^(n-1) …

**example Code**

```cpp
double X[15] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
double V[15] = {9.7, 8.1, 6.6, 5.1,4.4, 3.7, 2.8, 2.4, 2.0, 1.6, 1.4, 1.1, 0.85, 0.69, 0.6};
Matrix Z3;
Matrix X_1 = arr2Mat(X, 1, 15);
Matrix V_1 = arr2Mat(V, 1, 15);

expFit(Z3,X_1,V_1,1);
```
