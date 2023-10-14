
/*
 - Provides standard functionality for Matricies and Vectors
*/

#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

class Matrix {

    public:

        double** self;

        int rows = 0, columns = 0;

        Matrix( int _rows, int _columns){

            srand( time(NULL) );

            rows = _rows;
            columns = _columns;

            self = new double*[rows];

            for(int r = 0; r < rows; r++)

                self[r] = new double[columns];

            fill(0);

        }

        ~Matrix(){

            for (int r = 0; r < rows; r++) 

                delete[] self[r];

            delete[] self;

        }

        void addScalar( double scalar ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] += scalar;

        }

        void multiplyScalar( double scalar ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] *= scalar;

        }

        void add( Matrix const* mat ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] += mat->self[r][c];
                
        }

        void fill( double val ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] = val;
                
        }

        void copy( Matrix const* mat ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] = mat->self[r][c];

        }

        void randomize( double min, double max ){

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < columns; c++)

                self[r][c] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

        }

        void print(){

            cout << "\n";

            for(int r = 0; r < rows; r++){

                for(int c = 0; c < columns; c++)

                    cout << self[r][c] << " ";

                cout << "\n";

            }

        }

};

class Vector {

    public:

        double* self;

        int length;

        Vector( int _length ){

            length = _length;

            self = new double[length];

            fill(0);

        };

        ~Vector(){

            delete[] self;

        }

        void addScalar( double scalar ){

            for(int i = 0; i < length; i++)

                self[i] += scalar;

        }

        void multiplyScalar( double scalar ){

            for(int i = 0; i < length; i++)

                self[i] *= scalar;

        }

        void add( Vector const* vect ){

            for(int i = 0; i < length; i++)

                self[i] += vect->self[i];

        }

        void matrixVectorProduct( Matrix const* mat, Vector const* vect ){

            double dp;

            for(int l1 = 0; l1 < length; l1++){

                dp = 0;

                for(int l2 = 0; l2 < vect->length; l2++)

                    dp += vect->self[l2] * mat->self[l1][l2];

                self[l1] = dp;

            }

        }

        void fill( double val ){

            for(int i = 0; i < length; i++)

                self[i] = val;

        }

        void copy( Vector const* vect ){

            for(int i = 0; i < length; i++)

                self[i] = vect->self[i];

        }

        void randomize( double min, double max ){

            for(int i = 0; i < length; i++)

                self[i] = (double)(rand()) / (RAND_MAX + 1) * (max - min) + min;

        }

        void nonLinear( double (*func)(double) ){

            for(int i = 0; i < length; i++)

                self[i] = func(self[i]);

        }

        void print(){

            cout << "\n";

            for(int i = 0; i < length; i++)

                cout << self[i] << "\n";

        }

};

double sigmoid( double val ){

    return 1 / (1 + exp(-val));

}