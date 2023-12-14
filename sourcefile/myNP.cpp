/***********************************/
/* Created          : 09-07-2023   */
/* Modified         : 14-12-2023   */
/* Author           : Lee ChanKeun */
/***********************************/

#include "myNP.hpp"
#include <math.h>

#define SQ(X) ((X)*(X))
#define TRA(X) ((X)*(X)*(X))



/***********************************/
/*           Operand               */
/***********************************/

extern double rad2deg(double _x){
	return _x * 180/PI;
}

extern double deg2rad(double _x){
	return _x * PI/180;
}

// power fuction
extern double power(double _x, int N){
	double out=1;
	if(N<0){
		printf("ERROR! : power \n\n");
		return -1;
	}
	if(_x == 0){
		return 0;
	}
	if(_x == 1 || N == 0){
		return 1;
	}
	for(int i=0; i<N; i++){
		out*=_x;
	}
	return out;
}

// factorial function
extern double factorial(int N){
	double y = 1;
	if(N<0){
		printf("ERROR : N must be more than 0 \n\n");
		return -1;
	}
	if(N==0 || N==1){
		return 1;
	}
	for (int k = 2; k <= N; k++){
		y*=k;
	}
	return y;
}

//print
extern void printVec(double* _vec, int _row){

	for (int i = 0; i < _row; i++)
		printf("Vector[%d] = %.2f \n", i, _vec[i]);
	printf("\n");
}

/***********************************/
/*         taylor series           */
/***********************************/


//  Taylor series approximation for sin(x) using pre-defined functions (input unit: [rad])
extern double sinTaylor(double _x){	
	int N_max = 10;
	double S_N = 0;			

	for (int k = 0; k < N_max; k++){
        S_N += ((power(-1, k)) * power(_x,(2*k+1))) / factorial(2*k+1);
    }
	
	return S_N;
}

// Taylor series approximation for sin(x) using pre-defined functions (input unit: [deg])
extern double sindTaylor(double _x){
	//_x = degree
	_x= deg2rad(_x);
    return sinTaylor(_x);
}

// Taylor series approximation for cos(x) using pre-defined functions (input unit: [rad])
extern double cosTaylor(double _x){
	int N_max = 10;
	double C_N = 0;

	for ( int k = 0; k < N_max; k++){
		C_N += (power((-1),k) * power(_x,2*k)) / factorial(2*k);
	}
	return C_N;
}

// Taylor series approximation for cos(x) using pre-defined functions (input unit: [rad])
extern double cosdTaylor(double _x){
	//_x == degree
	_x= deg2rad(_x);
	return cosTaylor(_x);
}

/***********************************/
/*         nonlinear methods       */
/***********************************/

extern double bisection(double func(double),float _a0, float _b0, float _tol) {
	// Initialization
	int k = 0;
	int Nmax = 100;
	float a = _a0;
	float b = _b0;
	float xn = 0;
	float ep = 1000;

	// Bisection 
	while (k<Nmax && ep>_tol){
		//exception
		if(func(a)*func(b)>=0){
            printf("Error of same signed result");
            return -1;
        }
		// Update xn as midpoint
        xn = (a+b)/2;
		
        // Check tolerance
		ep = fabs(func(xn));
        
        if (func(xn)==0){
			break;
		}
		// Update range a, b
		else if (func(xn)*func(a)<0){
            b=xn;
        }
        else{
            a=xn;
        }
		k++;
		printf("k:%d \t", k);
		printf("Xn(k): %f \t", xn);
		printf("Tol: %.8f\n", ep);
	} 
	return xn;
}

extern double newtonRaphson(double func(double), double dfunc(double), double _x0, double _tol){
	double xn = _x0;
	double ep = 10000;
	int Nmax = 1000;
	int k = 0;
	double h = 0;
	printf("k:%d \t", k);
	printf("X(k): %f \t", xn);
	printf("Tol: %.10f\n", ep);
	do {
		if (dfunc(xn) == 0){
			printf("[ERROR] dfunc == 0 !!\n");
			break;
		}
		else{
			h=-func(xn)/dfunc(xn);   

			xn=xn+h;
            // check tolerance
            ep=fabs(func(xn));
			k++;

			printf("k:%d \t", k);
			printf("X(k): %f \t", xn);
			printf("Tol: %.10f\n", ep);

		}
	} while (k < Nmax && ep > _tol);
	return xn;
}


extern double secant(double func(double), double _x0, double _x1, double _tol){
	double xn0 = _x0;
    double xn1 = _x1;
    double xn2;
    double ep = 100;
    int Nmax = 1000;
    int k = 0;
    double h = 0;
	
	do {
        if ((func(xn1)-func(xn2)) == 0) {
            std::cerr << "[ERROR] dfunc == 0 !!" << std::endl;
            break;
        } else {
            h = -func(xn1) * (xn1-xn0) / (func(xn1) - func(xn0));
            xn2 = xn1 + h;
            
            ep = std::fabs(func(xn2));
            k++;

            std::cout << "k: " << k << "\t";
            std::cout << "X(k): " << xn2 << "\t";
            std::cout << "Tol: "  << ep << std::endl;
            
            xn0 = xn1;
            xn1 = xn2;
        }
    } while (k < Nmax && ep > _tol);
	return xn2;
}


/***********************************/
/*         distribution            */
/***********************************/

extern void gradient1D(double x[], double y[], double dydx[], int m) {	
	if (m<3) {
		printf("ERROR: length of data must be more than 2\n");
		return;
	}
	// Calculate h
	double h = x[1] - x[0];
	// first point
	dydx[0] = (-3*y[0] + 4*y[1] - y[2]) / (2*h);

	// mid points
	// Two-Point Central  O(h^2)
	for (int i = 1; i < m - 1; i++) {
		dydx[i] = (y[i+1] - y[i-1]) / (2*h); // [TO-DO] YOUR CODE GOES HERE
	}
	// For end point
	// Three-Point BWD  O(h^2). Need to use last 2 points
	dydx[m-1] = (3*y[m-1] - 4*y[m-2] + y[m-3]) / (2*h);
}


extern void acceleration(double x[ ], double y[ ], double dydx[ ], int m){
	
	if (m<4) {
		printf("ERROR: length of data must be more than 2\n");
		return;
	}
	double h = power(x[1] - x[0],2);

	// First Point:	Four-point forward difference 
	dydx[0] = (2*y[0] - 5*y[1] + 4*y[2] - y[3]) / (h);
	// Mid Points:	Three-point central difference
	for (int i = 1; i < m - 1; i++) {
		dydx[i] = (y[i+1] - 2*y[i] + y[i-1]) / (h); 
	}
	// End Point:	Four-point backward difference
	dydx[m-1] = (-y[m-4] + 4*y[m-3] - 5*y[m-2] + 2*y[m-1]) / (h);
}
 


// Truncation error should be O(h^2) 
extern void gradientFunc(double func(const double x), double x[], double dydx[], int m) {
	double* y;

	y = (double*)malloc(sizeof(double) * m);
	for (int i = 0; i < m; i++) {
		y[i] = func(x[i]);
	}
	//choose type
	// gradient1D(x, y,dydx,m);
	acceleration(x, y,dydx,m);
	
	free(y);
}

/***********************************/
/*         integration            */
/***********************************/

extern double trapz(double x[], double y[], int m) {
	//m = N
	double I = 0;

	for (int i = 1; i < m; i++){
		double dx = x[i] - x[i - 1];
        double y_avg = (y[i] + y[i - 1]) / 2.0;
        I += dx * y_avg;
	}
	return I;
}

extern double simpson13(double x[], double y[], int m){
	
	if (m<2) {
		printf("ERROR: length of data must be more than 2\n");
		return 0 ;
	}
	double h = x[1]-x[0];
	double Is = 0;
	for (int i = 0; i<m ; i++){
		if(i==0){
			Is+= y[0]+y[m];
		}
		else if(i%2 == 1){
			Is+= 4*(y[i]);
		}else{
			Is+= 2*(y[i]);
		}
	}
	Is = h*Is/3;
	return Is;
}

extern double simpson38(double x[], double y[], int m){
	if (m<3) {
		printf("ERROR: length of data must be more than 3\n");
		return 0 ;
	}
	double h = x[1]-x[0];
	double Is = 0;
	for (int i = 0; i<m ; i++){
		if(i==0){
			Is+= y[0]+y[m];
		}
		else if(i%3 == 0){
			Is+= 2*(y[i]);
		}else if(i%3 == 1){
			Is+= 3*(y[i]);
		}else if(i%3 == 2){
			Is+= 3*(y[i]);
		}
	}
	Is = (3*h*Is)/8;
	return Is;
}

//odd
extern double integral(double func(const double), double a, double b, int n){
	if (n<=0){
		printf("ERROR: n must be positive integer");
		return -1;
	}
	if(n%2==1){
		n++;
	}
	double h = (b-a)/n;
	double result;

	//x0
	result = func(a) - func(b);
	for(int i=1; i<n; i+=2){
		result += 4.*func(a+h*i) + 2.*func(a+h*(i+1));
	}
	result *= h/3.;
	
	return result;
}

/***********************************/
/*         interpolation           */
/***********************************/
//xq 일때의 값을 예측
//n = order
//m = 데이터 개수
extern double lagrageInterpolation(double* x, double* y, double xq, int m, int order){
	if (order < 1){
		printf("Order is too little.\n\n");
		return 0;
	}
	else if (order > m - 1){
		printf("Order is should be smaller Array size.\n\n");
		return 0;
	}

	int min, max;
	(order % 2 == 1) ? (min = (order - 1) / 2) : (min = order / 2);
	max = order - min;

	double yq = 0;
	double fn;
	
	for (int i = min; i < m - max; i++){
		if ((x[i - min] <= xq) && (x[i + max] >= xq)){

			for (int j = 0; j <= order; j++){
				
				fn = y[i + j];
				for (int k = 0; k <= order; k++){
					if (j == k) continue;
					fn *= (xq - x[i + k]) / (x[i + j] - x[i + k]);
				}
				if(fn > 1e-6)
					yq += fn;
			}

			return yq;
		}
	}
	printf("Error : Not have xq in data range\n\n");

	return 0;
}

extern double newtonInterpolation(double* x, double* y, double xq, int m, int order){
	if (order < 1){
		printf("Order is too little.\n\n");
		return 0;
	}
	
	else if (order > m - 1){
		printf("Order is should be smaller Array size.\n\n");
		return 0;
	}

	int min, max;
	(order % 2 == 1) ? (min = (order - 1) / 2) : (min = order / 2);
	max = order - min;

	double yq = 0;
	double* c;
	c = (double*)malloc(sizeof(double) * m);

	if (c == NULL)
	{
		printf("Error : Fail assignment memory.\n\n");

		return 0;
	}

	double fn = 0;
	double fc = 0;

	for (int i = min; i < m - max; i++){
		if ((x[i - min] <= xq) && (x[i + max] >= xq)){
			for (int j = 0; j <= order; j++){
				c[j] = y[i + j];
				for (int k = 0; k < j; k++){
					fc = c[k];
					for (int l = 0; l < k; l++)
						fc *= (x[i + k] - x[i + l]);
					c[j] -= fc;
				}
				for (int k = 0; k < j; k++)
					c[j] /= (x[i + j] - x[i + k]);
			}
			for (int j = 0; j <= order; j++){
				fn = c[j];
				for (int k = 0; k < j; k++)
					fn *= (xq - x[i + k]);
				if(fn > 1e-6)
					yq += fn;
			}
			free(c);
			return yq;
		}
	}
	printf("Error : Not have xq in data range\n\n");
	free(c);
	return 0;
}

extern double linearSpline(double* x, double* y, int n, double xq) {
    if (n < 2) {
        printf("Error : not enough data\n\n");
        return 0.0;
    }
    for (int i = 0; i < n - 1; i++) {
        if (xq >= x[i] && xq <= x[i + 1]) {
            // 주어진 x 값이 두 데이터 포인트 사이에 있는 경우
            double x0 = x[i];
            double x1 = x[i + 1];
            double y0 = y[i];
            double y1 = y[i + 1];

            return y0*(xq-x1)/(x0-x1) + y1*(xq - x0)/ (x1 - x0);
        }
    }

    printf("Error : out of data range\n\n");
    return 0.0;
}


/***********************************/
/*               ODE               */
/***********************************/

//2nd order Runge-Kutta method 
//y: 1-D array for output y(t). The length should be predefined and fixed.
// myfunc(y,t)=dy/dt 
// t0,tf, h: start time, end time and time intervals, respectively. 
// default value is alpha=1 
extern void odeRK2(double myfunc(const double, const double), double y[], double t0, double tf, double h, double y0){
	// Number of steps
	// or length of y
	// int((tf-t0)/h) unstable
    int num_steps = static_cast<int>((tf - t0) / h);
	
    // Initialize variables
    double t = t0;
    y[0] = y0;

    // Runge-Kutta method
    for (int i = 0; i < num_steps; i++) {
        double k1 = h * myfunc(t, y[i]);
        double k2 = h * myfunc(t + h, y[i] + k1);

        y[i + 1] = y[i] + 0.5 * (k1 + k2);
        t += h;
    }
	return;
}

//3rd order Runge-Kutta method 
extern void ode(double myfunc(const double t, const double y), double y[], double t0, double tf, double h, double y0) {
    // Number of steps
	// or length of y
	// int((tf-t0)/h) unstable
    int num_steps = static_cast<int>((tf - t0) / h);

    // Initialize variables
    double t = t0;
    y[0] = y0;

    // Runge-Kutta method (RK3)
    for (int i = 0; i < num_steps; i++) {
        double k1 = h * myfunc(t, y[i]);
        double k2 = h * myfunc(t + 0.5 * h, y[i] + 0.5 * k1 * h);
        double k3 = h * myfunc(t + h, y[i] - k1*h + 2.0 * k2 * h);

        y[i + 1] = y[i] + (k1 + 4.0 * k2 + k3) / 6.0;
        t += h;
    }
}

// 2nd order sys RK2
// void mckfunc(const double t, const double y[], double dydt[])
extern void sysRK2(void myfunc(const double, double[], double[]), double y[], double z[], double a, double b, double h, double yINI, double zINI){

	// Number of steps 
    int num_steps = static_cast<int>((b - a) / h);
	
    // Initialize variables
	double t = a;
    y[0] = yINI;
	z[0] = zINI;
	double ky1, kz1, ky2, kz2;

	double dydt[2];
	double tmp[2];

    // Runge-Kutta method
    for (int i = 0; i < num_steps; i++) {
		tmp[0] = y[i];
		tmp[1] = z[i];

		myfunc(t, tmp, dydt);
		
        ky1 =  h * dydt[0];
        kz1 =  h * dydt[1];

		tmp[0] += ky1;
		tmp[1] += kz1;

		myfunc(t+h, tmp, dydt);
		ky2 = h * dydt[0];
        kz2 = h * dydt[1];

        y[i+1] = y[i] + 0.5 * (ky1 + ky2);
		z[i+1] = z[i] + 0.5 * (kz1 + kz2);
        t += h;
    }

	return;
}