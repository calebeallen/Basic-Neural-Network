
#include "Linear.cpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

class NeuralNetwork{

    public:

        vector<Matrix*> W, wGradient;

        vector<Vector*> A, aGradient, B, bGradient, zGradient;

        Vector* Y;

        int layers, tCount = 0;

        double lRate, initialC;

        NeuralNetwork(int _layers, double _lRate, int* nodes){

            layers = _layers;

            lRate = _lRate;

            //Create activation vectors for each layer
            for(int i = 0; i < layers; i++){

                if(i != 0){

                    B.push_back( new Vector(nodes[i]) );

                    bGradient.push_back( new Vector(nodes[i]) );

                    zGradient.push_back( new Vector(nodes[i]) );

                    aGradient.push_back( new Vector(nodes[i]) );

                }

                A.push_back( new Vector(nodes[i]) );

            }

            //Create and randomize weight matricies
            for(int i = 0; i < layers - 1; i++){

                W.push_back( new Matrix( nodes[i + 1], nodes[i] ) );

                W[i]->randomize(-1,1);

                wGradient.push_back( new Matrix( nodes[i + 1], nodes[i] ) );

            }

            //Create y vector
            Y = new Vector( nodes[layers-1] );
            
        }

        void setInput( Vector* input, Vector* ans ){

            A[0]->copy( input );

            Y->copy( ans );

        }

        void run(){

            for(int i = 0; i < layers - 1; i++){

                //Calculate activation by matrix vector product of previous activations and weights
                A[i+1]->matrixVectorProduct( W[i], A[i] );

                A[i+1]->add(B[i]);

                //Apply nonlinear function
                A[i+1]->nonLinear( &sigmoid );

            }

            if(initialC == 0)

                initialC = cost();

        }

        void train(){

            run();

            double dCdaK;

            /* Back Propagation */

            for(int i = 0; i < Y->length; i++)

                //Compute partial derivative of cost with respect to output layer activations
                aGradient[layers - 2]->self[i] = 2 * ( A[layers - 1]->self[i] - Y->self[i] );

            //Work backwards through layers
            for(int l = layers - 2; l >= 0; l--){

                for(int j = 0; j < A[l + 1]->length; j++)

                    // ∂C/∂Zj = ∂C∂Aj * σ'(Zj)
                    zGradient[l]->self[j] = ( A[l + 1]->self[j] - pow( A[l + 1]->self[j], 2 ) ) * aGradient[l]->self[j];

                for(int k = 0; k < A[l]->length; k++){

                    if(l > 0)
                    
                        aGradient[l - 1]->self[k] = 0;

                    for(int j = 0; j < A[l + 1]->length; j++){

                        // ∂C/∂Wjk = ∂C/∂Zj * ∂Zj/∂Wjk = ∂C/∂Zj * Ak
                        wGradient[l]->self[j][k] += zGradient[l]->self[j] * A[l]->self[k];

                        if(l > 0)
                        // ∂C/∂Ak = Σj ∂C/∂Zj * ∂Zj/∂AK = Σj ∂C/∂Zj * Wjk
                        aGradient[l - 1]->self[k] += zGradient[l]->self[j] * W[l]->self[j][k];

                    }

                    //aGradient[l]->self[k] = dCdaK;

                }

                // ∂C/∂Bj = ∂C/∂Zj * ∂Zj/∂Bj = ∂C/∂Zj
                bGradient[l]->add(zGradient[l]);

            }

            tCount++;

        }

        void applyGradient(){

            //Compute gradient scalar, which is the negative learning rate divided by the amount of training examples used.
            //This gives the average negative gradient multiplied by the learning rate.
            double scalar = -lRate / tCount;

            for(int i = 0; i < layers - 1; i++){

                //divide gradient by total training examples, apply learning rate
                wGradient[i]->multiplyScalar( scalar );

                bGradient[i]->multiplyScalar( scalar );

                //add negative gradient to weights and biases
                W[i]->add(wGradient[i]);

                B[i]->add(bGradient[i]);

                //reset gradients
                wGradient[i]->fill(0);

                bGradient[i]->fill(0);
                
            }

            tCount = 0;

        }

        double cost(){

            double sum = 0;

            for(int i = 0; i < Y->length; i++)

                sum += pow( A[layers - 1]->self[i] - Y->self[i], 2 );

            return sum;

        }

        int predict(){

            double max = 0;

            int ind;

            for(int i = 0; i < A[layers - 1]->length; i++)

                if(A[layers - 1]->self[i] > max){

                    max = A[layers - 1]->self[i];

                    ind = i;

                }

            return ind;

        }

        void print(){

            for(int i = 0; i < layers - 1; i++){

                cout<< "Activiation " << i << "\n";
                A[i]->print();

                cout<< "Weights " << i << "\n";
                W[i]->print();

            }

            cout<< "Output Layer \n";
            A[layers - 1]->print();

            cout<< "Y Vector \n";
            Y->print();
            
            cout<< "Cost \n" << cost() << "\n";

        }

        void printOutput(){

            cout<<"\n\n";

            for(int i = 0; i < A[layers - 1]->length; i++){

                cout << i << " -> " << (int)(A[layers - 1]->self[i] * 100) << "%  confident" << "\n";

            }

        }

        void save(string dir){

            ofstream file;

            string arr;

            file.open( dir, ios::trunc );

            for(int i = 0; i < layers - 1; i++){

                arr = "";

                for(int r = 0; r < W[i]->rows; r++){

                    for(int c = 0; c < W[i]->columns; c++){

                        arr += to_string(W[i]->self[r][c]);

                        arr += ",";

                    }

                    arr += "/,";

                }

                arr += "\n";

                file << arr;

                arr = "";

                for(int k = 0; k < B[i]->length; k++){

                    arr += to_string(B[i]->self[k]);

                    arr += ',';

                }

                arr += "\n";

                file << arr;

            }

            file.close();

        }

        void load(string dir){

            ifstream file(dir);

            string line, val;

            int i = 0;

            if(file.is_open())

                while(getline(file, line)){

                    val = "";

                    if(i % 2 == 0){

                        for(int k = 0, r = 0, c = 0; k < line.length(); k++){

                            if(line[k] == ','){

                                if(val == "/"){

                                    r++;

                                    c = 0;
                                    
                                }else{

                                    W[i / 2]->self[r][c] = stod(val);

                                    c++;

                                }

                                val = "";

                            }else

                                val += line[k];

                        }

                    } else {

                        for(int k = 0, j = 0; k < line.length(); k++){

                            if(line[k] == ','){

                                B[i / 2]->self[j] = stod(val);

                                j++;

                                val = "";

                            }else

                                val += line[k];

                        }

                    }

                    i++;
                    
                }

            file.close();

        }

};
