package network
import scala.util.Random
import breeze.linalg._
import breeze.numerics._

object Utils {
    
    def initializeWeights(layers: Array[Int]): Array[DenseMatrix[Double]] = {
        val weights: Array[DenseMatrix[Double]] = Array.ofDim[DenseMatrix[Double]](layers.length-1);
        for(i<-0 to layers.length-2) {
            weights(i) = DenseMatrix.rand(layers(i+1), layers(i))*0.01;
        }
        return weights;
    }

    def initializeBiases(layers: Array[Int]): Array[DenseVector[Double]] = {
        val biases: Array[DenseVector[Double]] = Array.ofDim[DenseVector[Double]](layers.length-1);
        for(i<-0 to layers.length-2) {
            biases(i) = DenseVector.rand[Double](layers(i+1))*0.01;
        }
        return biases;
    }

    def sigmoid(X: DenseMatrix[Double]): DenseMatrix[Double] = {
        1.0 / (exp(-X) + 1.0)
    }

    def dsigmoid(X: DenseMatrix[Double]): DenseMatrix[Double] = {
        X *:* (1.0 - X);
    }

    def relu(X: DenseMatrix[Double]): DenseMatrix[Double] = {
        max(X, 0.0);
    }
    
    def relu_derivative(X: DenseMatrix[Double]): DenseMatrix[Double] = {
        X.map { x =>
            if (x > 0.0) 1.0 else 0.0
        }
    }

    def softmax(X: DenseMatrix[Double]): DenseMatrix[Double] = {
        // Step 1: Find the maximum value in the matrix
        val maxVal = max(X)
        println(maxVal)
        // Step 2: Compute the logsumexp
        val short = exp(X - maxVal)
        println("Short");
        println(short);
        val sumExp = sum(short(*,::))
        println("\nSumExp");
        println(sumExp);
        val logsumexp = log(sumExp) + maxVal
        println("\n" + logsumexp)
        val soft = exp(X(::,*) - logsumexp)
        println("\n" + soft)
        return soft;
    }

}