package network

import breeze.linalg._
import breeze.numerics._
import breeze.linalg.LU.primitive.LU_DM_Impl_Double
import breeze.linalg.operators._
import breeze.math.Semiring

class Network(val layers: Array[Int], val weights: Array[DenseMatrix[Double]], val biases: Array[DenseVector[Double]]) {

    val activations: Array[DenseMatrix[Double]] = Array.ofDim[DenseMatrix[Double]](layers.length);
    val gradients: Array[DenseMatrix[Double]] = Array.ofDim[DenseMatrix[Double]](layers.length-1);
    val grad_biases: Array[DenseVector[Double]] = Array.ofDim[DenseVector[Double]](layers.length-1);
    var final_activation: DenseMatrix[Double] => DenseMatrix[Double] = Utils.sigmoid
    var deltas: Array[DenseMatrix[Double]] = Array.ofDim[DenseMatrix[Double]](layers.length-1);
    var activation_fn: DenseMatrix[Double] => DenseMatrix[Double] = Utils.sigmoid;
    var dactivations: DenseMatrix[Double] => DenseMatrix[Double] = Utils.dsigmoid;
    var regularizer: Double = 0.0;
    var regularization: Boolean = false;

    def this(layers: Array[Int], 
        weights: Array[DenseMatrix[Double]], 
        biases: Array[DenseVector[Double]], 
        final_activation: DenseMatrix[Double] => DenseMatrix[Double],
        activation_fn: DenseMatrix[Double] => DenseMatrix[Double],
        derivative_fn: DenseMatrix[Double] => DenseMatrix[Double],
    ) = {
        this(layers, weights, biases);
        this.final_activation = final_activation;
        this.activation_fn = activation_fn;
        this.dactivations = derivative_fn
    }

        def this(layers: Array[Int], 
        weights: Array[DenseMatrix[Double]], 
        biases: Array[DenseVector[Double]], 
        final_activation: DenseMatrix[Double] => DenseMatrix[Double],
        activation_fn: DenseMatrix[Double] => DenseMatrix[Double],
        derivative_fn: DenseMatrix[Double] => DenseMatrix[Double],
        regularizer: Double
    ) = {
        this(layers, weights, biases);
        this.final_activation = final_activation;
        this.activation_fn = activation_fn;
        this.dactivations = derivative_fn;
        this.regularizer = regularizer;
        this.regularization = true;
    }


    def forward_pass(inputs: DenseMatrix[Double]) = {
        activations(0) = inputs;
        for(i<-1 to layers.length-2) {
            activations(i) = activations(i-1) * weights(i-1).t;
            activations(i) = activations(i)(*, ::) + biases(i-1);
            activations(i) = this.activation_fn(activations(i));
        }
        activations(layers.length-1) = activations(layers.length-2) * weights(layers.length-2).t;
        activations(layers.length-1) = activations(layers.length-1)(*, ::) + biases(layers.length-2);
        activations(layers.length-1) = this.final_activation(activations(layers.length-1));
    }

    def backprop(outputs: DenseMatrix[Double]) = {
        var length = deltas.length - 1;
        deltas(length) = activations(length+1) - outputs;
        grad_biases(length) = sum(deltas(length)(::, *)).t;
        for(i<- (length - 1) to 0 by -1) {
            gradients(i+1) = (deltas(i+1).t * activations(i+1))
            deltas(i) = deltas(i+1) * weights(i+1);
            deltas(i) = deltas(i) *:* this.dactivations(activations(i+1));
            grad_biases(i) = sum(deltas(i)(::, *)).t;
        }
        gradients(0) = (deltas(0).t * activations(0));
        grad_biases(0) = sum(deltas(0)(::, *)).t;
        for(i <- (length) to 0 by -1) {
            if(this.regularization) {
                var mask = this.weights(i) * this.regularizer;
                gradients(i) += mask;
            }
            gradients(i) = gradients(i) / activations(0).rows.toDouble
            grad_biases(i) = grad_biases(i) / activations(0).rows.toDouble
        }
    }

    def updateWeights(learningRate: Double) = {
        for(i <- 0 to weights.length-1) {
            weights(i) = weights(i) - gradients(i) * learningRate;
            biases(i) = biases(i) - grad_biases(i) * learningRate;
        }
    }

}