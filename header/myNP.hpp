/***********************************/
/* Created          : 09-07-2023   */
/* Modified         : 14-12-2023   */
/* Author           : Lee ChanKeun */
/***********************************/

#ifndef		_MY_NP_H	// use either (#pragma once) or  (#ifndef ...#endif)
#define		_MY_NP_H
#define		PI		3.14159265358979323846264338327950288419716939937510582

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>


/***********************************/
/*           Operand               */
/***********************************/

// degree and Radian
extern double rad2deg(double x);
extern double deg2rad(double x);

//operator
extern double power(double _x, int N);
extern double factorial(int _x);

//print
extern void printVec(double* _vec, int _row);

/***********************************/
/*         taylor series           */
/***********************************/

extern double sinTaylor(double _x);
extern double sindTaylor(double _x);
extern double cosTaylor(double _x);
extern double cosdTaylor(double _x);

/***********************************/
/*         nonlinear               */
/***********************************/

extern double bisection(double func(double), float _a0, float _b0, float _tol);
extern double newtonRaphson(double func(double), double dfunc(double), double _x0, double _tol);
extern double secant(double func(double), double _x0, double _x1, double _tol);


/***********************************/
/*         distribution            */
/***********************************/

extern void gradient1D(double _x[], double _y[], double dydx[], int m);
extern void acceleration(double x[ ], double y[ ], double dydx[ ], int m);
extern void gradientFunc(double func(const double x), double x[], double dydx[], int m);


/***********************************/
/*         Integration            */
/***********************************/
extern double trapz(double x[], double y[], int m);
extern double simpson13(double _x[], double _y[], int m);
extern double simpson38(double x[], double y[], int m);
extern double integral(double func(const double), double a, double b, int m);


/***********************************/
/*         Interpolation           */
/***********************************/
extern double lagrageInterpolation(double* x, double* y, double xq, int m, int order);
extern double newtonInterpolation(double* x, double* y, double xq, int m, int order);
extern double linearSpline(double* x, double* y, int n, double xq);

/***********************************/
/*               ODE               */
/***********************************/

//2nd order Runge-Kutta method 
extern void odeRK2(double myfunc(const double, const double), double y[], double t0, double tf, double h, double y0);
//3rd order Runge-Kutta method 
extern void ode(double myfunc(const double t, const double y), double y[], double t0, double tf, double h, double y0);
//2nd order sys RK2
extern void sysRK2(void myfunc(const double, double[],  double[]), double y[], double z[], double a, double b, double h, double yINI, double zINI);


#endif