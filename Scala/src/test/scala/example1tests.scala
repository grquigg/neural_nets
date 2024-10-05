package network
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfter
import breeze.linalg._
import breeze.linalg.NumericOps.Arrays.ArrayIsNumericOps

class Example1Spec extends AnyFlatSpec with Matchers with BeforeAndAfter {

  var layers: Array[Int] = _;
  var weights: Array[DenseMatrix[Double]] = _;
  var biases: Array[DenseVector[Double]] = _;
  var model: Network = _;
  var inputs: DenseMatrix[Double] = DenseMatrix(
      (0.13),
      (0.42)
    );
  var outputs: DenseMatrix[Double] = _;
  // Example of a fixture

  def approximatelyEqualMatrices(m1: DenseMatrix[Double], m2: DenseMatrix[Double], tolerance: Double = 1e-4): Boolean = {
    if (m1.rows != m2.rows || m1.cols != m2.cols) return false
    (m1 - m2).toArray.forall(_.abs < tolerance)
  }

  before {
    layers = Array(1,2,1);
    weights = Array(
      DenseMatrix((0.1), (0.2)),
      DenseMatrix((0.5, 0.6))
    );
    biases = Array(
      DenseVector(0.4, 0.3),
      DenseVector(0.7)
    );
    model = Network(layers, weights, biases);
    outputs = DenseMatrix(
      (0.9),
      (0.23)
    );
  }
  after {}
  "Forward activations" should "be equal to what we would expect" in {
      model.forward_pass(inputs);
      assert(approximatelyEqualMatrices(model.activations(1), DenseMatrix((0.60181,0.58079), (0.60874,0.59484))))
      assert(approximatelyEqualMatrices(model.activations(2), DenseMatrix((0.79403), (0.79597))))
  }


  "Softmax function" should "be equal to 0.5 for a 2x2 matrix of 1s" in {
      val probs = Utils.softmax(DenseMatrix((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)));
      assert(approximatelyEqualMatrices(probs, DenseMatrix((0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25))));
  }

  "Softmax function" should "be 1.0 for a 2x1 matrix" in {
      val probs = Utils.softmax(DenseMatrix((1.0), (1.0)));
      assert(approximatelyEqualMatrices(probs, DenseMatrix((1.0), (1.0))));
  }

  "Forward activations" should "be correct for a sigmoid network with a final softmax layer" in {
      model.final_activation = Utils.softmax
      model.forward_pass(inputs);
      assert(approximatelyEqualMatrices(model.activations(2), DenseMatrix((1.0), (1.0))))
  }

  "Deltas and gradients" should "be correct for the backprop algorithm" in {
      model.forward_pass(inputs);
      model.backprop(outputs);
      assert(model.activations.length == 3);
      assert(approximatelyEqualMatrices(model.deltas(1), DenseMatrix((-0.10597257),(0.56596607))));
      assert(approximatelyEqualMatrices(model.deltas(0), DenseMatrix((-0.01270,-0.01548),(0.06740,0.08184))));
      assert(model.gradients(1).rows == 1);
      assert(model.gradients(1).cols == 2);
      assert(model.gradients(0).rows == 2);
      assert(model.gradients(0).cols == 1);
      assert(approximatelyEqualMatrices(model.gradients(1), DenseMatrix((0.14037,0.13756))));
      assert(approximatelyEqualMatrices(model.gradients(0), DenseMatrix((0.01333), (0.01618))));
  }

  "Forward prop" should "be correct for relu activations" in {
      model.activation_fn = Utils.relu;
      model.forward_pass(inputs);
      assert(approximatelyEqualMatrices(model.activations(1), DenseMatrix((0.413, 0.326), (0.442, 0.384))))
      assert(approximatelyEqualMatrices(model.activations(2), DenseMatrix((0.75065), (0.75976654))))
  }

  "Deltas and gradients" should "be correct for the backprop algorithm with relu activations" in {
      model.activation_fn = Utils.relu;
      model.dactivations = Utils.relu_derivative;
      model.forward_pass(inputs);
      model.backprop(outputs);
      assert(approximatelyEqualMatrices(model.deltas(1), DenseMatrix((-0.14935), (0.5298))));
      assert(approximatelyEqualMatrices(model.deltas(0), DenseMatrix((-0.0747, -0.08961), (0.2649, 0.31788))));
      assert(model.gradients(1).rows == 1);
      assert(model.gradients(1).cols == 2);
      assert(model.gradients(0).rows == 2);
      assert(model.gradients(0).cols == 1);
      assert(approximatelyEqualMatrices(model.gradients(1), DenseMatrix((0.0862,0.07735))));
      assert(approximatelyEqualMatrices(model.gradients(0), DenseMatrix((0.05077172),(0.06092607))));
  }

  "Deltas and gradients" should "be correct for the backprop algorithm with sigmoid and regularization" in {
      model.activation_fn = Utils.sigmoid;
      model.regularization = true;
  }
}