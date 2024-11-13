package network

import java.io.{File, FileInputStream, BufferedInputStream}
import java.nio.ByteBuffer
import java.nio.ByteOrder
import scala.math.{BigInt}
import spire.implicits._
import scala.util.Random
import breeze.linalg._

object Main {

    def readUByteFile(filePath: String): DenseMatrix[Short] = {
        val file = new File(filePath)
        //Scala doesn't support unsigned integers natively, so there's need to be creative here
        val buffer = new Array[Byte](file.length.toInt)
        val bis = new BufferedInputStream(new FileInputStream(file))
        try {
            bis.read(buffer)
        } finally {
            bis.close()
        }
        //first two bytes are magic 
        val dims = buffer(3)
        val dimensions = new Array[BigInt](dims)
        for(i<-0 to dims-1) {
            dimensions(i) = BigInt(buffer.slice(4+(4*i), (4+(4*(i+1)))))
        }
        var flatten = BigInt(1);
        for(i<-1 to dims-1) {
            flatten *= dimensions(i);
        }
        var buffer_ptr = 4 + (dims * 4);
        //the data in both files is of the format unsigned byte
        val arr = DenseMatrix.zeros[Short](dimensions(0).intValue(), flatten.intValue())
        for(i<-0 to dimensions(0).intValue()-1) {
            for(j<-0 to flatten.intValue()-1) {
                arr(i, j) = BigInt(buffer(buffer_ptr)).toShort
                if(arr(i, j) < 0) {
                    arr(i, j) = (arr(i, j) + 256).toShort;
                }
                buffer_ptr += 1;
            }
        }
        return arr;
        
    }

    def convertIntToFloatArray(arr: DenseMatrix[Short]): DenseMatrix[Double] = {
        return arr.map(_.toDouble);
    }

    def convertInputsToOneHot(outputs: DenseMatrix[Short]): DenseMatrix[Double] = {
        var output: DenseMatrix[Double] = DenseMatrix.zeros[Double](outputs.rows, 10);
        for(i<-0 to outputs.rows-1) {
            output(i, outputs(i, 0).toInt) = 1.0f;
        }
        return output;
    }

    def main(args: Array[String]) = {
        //add rest of args as well
        val epochs = 100;
        val BATCH_SIZE = 4000;
        var learningRate = 0.001f;
        val train_data_path: String = args(0);
        val train_label_path: String = args(1);
        //read in data
        var inputs: DenseMatrix[Double] = convertIntToFloatArray(readUByteFile(train_data_path));
        inputs = inputs / 255.0;
        var outputs: DenseMatrix[Short] = readUByteFile(train_label_path);
        //set layer params
        var layers: Array[Int] = Array(784, 64, 10);
        var weights: Array[DenseMatrix[Double]] = Utils.initializeWeights(layers);
        var biases: Array[DenseVector[Double]] = Utils.initializeBiases(layers);
        var output_one_hot: DenseMatrix[Double] = convertInputsToOneHot(outputs);
        var model: Network = new Network(layers, weights, biases);
        model.final_activation = Utils.softmax;
        model.activation_fn = Utils.relu;
        model.dactivations = Utils.relu_derivative;
        var accuracy: Double = 0.0;
        var totalCorrect: Int = 0;
        var loss: Double = 0.0;
        println(inputs.rows)
        for(i <- 0 until epochs) {
            totalCorrect = 0;
            loss = 0.0;
            for(i <- 0 until inputs.rows by BATCH_SIZE) {
                model.forward_pass(inputs(i until i+BATCH_SIZE, ::).copy);
                loss = loss + Utils.crossEntropyLoss(output_one_hot(i until i+BATCH_SIZE, ::), model.activations(layers.length-1));
                
                totalCorrect = totalCorrect + Utils.computeCorrect(output_one_hot(i until i+BATCH_SIZE, ::), model.activations(layers.length-1));
                
                model.backprop(output_one_hot(i until i+BATCH_SIZE, ::));
                model.updateWeights(learningRate);
            }
            println(loss);
            accuracy = totalCorrect / inputs.rows.toDouble;
            println(String.format("Accuracy: %.5f%%", accuracy*100))
        }
    }
}