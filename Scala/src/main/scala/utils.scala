package network
import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import scala.util.boundary, boundary.break

object Utils {
    Rand.generator.setSeed(42);

    def initializeWeights(layers: Array[Int]): Array[DenseMatrix[Double]] = {
        val weights: Array[DenseMatrix[Double]] = Array.ofDim[DenseMatrix[Double]](layers.length-1);
        for(i<-0 to layers.length-2) {
            weights(i) = DenseMatrix.rand(layers(i+1), layers(i), Rand.gaussian)*0.01;
        }
        return weights;
    }

    def initializeBiases(layers: Array[Int]): Array[DenseVector[Double]] = {
        val biases: Array[DenseVector[Double]] = Array.ofDim[DenseVector[Double]](layers.length-1);
        for(i<-0 to layers.length-2) {
            biases(i) = DenseVector.rand[Double](layers(i+1), Rand.gaussian)*0.01;
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
        // Step 2: Compute the logsumexp
        val short = exp(X - maxVal)
        val sumExp = sum(short(*,::))
        val logsumexp = log(sumExp) + maxVal
        val soft = exp(X(::,*) - logsumexp)
        return soft;
    }

    def crossEntropyLoss(expected: DenseMatrix[Double], actual: DenseMatrix[Double]): Double = {
        val eplison = 1e-15;
        val y_pred = clip(actual, eplison, 1 - eplison);
        val loss = (0 until actual.rows).map { i =>
            val trueRow = expected(i, ::).t
            val predRow = y_pred(i, ::).t
            -sum(trueRow *:* log(predRow))
        }

        val averageLoss = loss.sum / expected.rows;
        return averageLoss;
    }

    def computeCorrect(expected: DenseMatrix[Double], actual: DenseMatrix[Double]): Int = {
        val predictions = (0 until expected.rows).map(i => {
            argmax(expected(i, ::));
        });

        val correctAns = (0 until actual.rows).map(i => {
            argmax(actual(i, ::));
        });
        var ans: Int = 0;
        val numCorrect = correctAns == predictions;
        for(i <- 0 until predictions.length) {
            if(correctAns(i) == predictions(i)) {
                ans+=1;
            }
        }
        return ans;
    }
}