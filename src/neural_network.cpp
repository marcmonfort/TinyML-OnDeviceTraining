#include <arduino.h>
#include "neural_network.h"
#include <math.h>


NeuralNetwork::NeuralNetwork() {

}



void NeuralNetwork::initWeights() {
    
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= InputNodes ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = float(random(100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    for( i = 0 ; i < OutputNodes ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
        ChangeOutputWeights[j][i] = 0.0 ;  
        Rando = float(random(100))/100;        
        OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }
}




void NeuralNetwork::forward(const float Input[]){


/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for (i = 0; i < HiddenNodes; i++) {
        Accum = HiddenWeights[InputNodes][i];
        for (j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j][i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for (i = 0; i < OutputNodes; i++) {
        Accum = OutputWeights[HiddenNodes][i];
        for (j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j][i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        // OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        // Error += 0.5 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
}




void NeuralNetwork::backward(const float Input[], const float Target[]){


// FORWARD

/******************************************************************
* Compute hidden layer activations
******************************************************************/

    for (i = 0; i < HiddenNodes; i++) {
        Accum = HiddenWeights[InputNodes][i];
        for (j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j][i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

/******************************************************************
* Compute output layer activations and calculate errors
******************************************************************/

    for (i = 0; i < OutputNodes; i++) {
        Accum = OutputWeights[HiddenNodes][i];
        for (j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * OutputWeights[j][i];
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        Error += 0.5 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }


// END FORWARD




/******************************************************************
* Backpropagate errors to hidden layer
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
            Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

/******************************************************************
* Update Inner-->Hidden Weights
******************************************************************/

    for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j][i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
            HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
    }

/******************************************************************
* Update Hidden-->Output Weights
******************************************************************/

    for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
            OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
    }
}


float* NeuralNetwork::get_output(){
    return Output;
}