package network
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.BeforeAndAfter
import breeze.linalg._
import breeze.linalg.NumericOps.Arrays.ArrayIsNumericOps

class Example2Spec extends AnyFlatSpec with Matchers with BeforeAndAfter {

  var layers: Array[Int] = _;
  var weights: Array[DenseMatrix[Double]] = _;
  var biases: Array[DenseVector[Double]] = _;
  var model: Network = _;
  var inputs: DenseMatrix[Double] = DenseMatrix(
      (0.32, 0.68),
      (0.83, 0.02)
    );
  var outputs: DenseMatrix[Double] = _;
  // Example of a fixture

  def approximatelyEqualMatrices(m1: DenseMatrix[Double], m2: DenseMatrix[Double], tolerance: Double = 1e-4): Boolean = {
    if (m1.rows != m2.rows || m1.cols != m2.cols) return false
    (m1 - m2).toArray.forall(_.abs < tolerance)
  }

  def approximatelyEqualVectors(v1: DenseVector[Double], v2: DenseVector[Double], tolerance: Double = 1e-4): Boolean = {
    if (v1.size != v2.size) return false
    (v1 - v2).toArray.forall(_.abs < tolerance)
  }
  before {
    layers = Array(2,4,3,2);
    weights = Array(
      DenseMatrix((0.15, 0.1, 0.19, 0.35), (0.4, 0.54, 0.42, 0.68)),
      DenseMatrix((0.67, 0.42, 0.56), (0.14, 0.2, 0.8), (0.96, 0.32, 0.69), (0.87, 0.89, 0.09)),
      DenseMatrix((0.87, 0.1), (0.42, 0.95), (0.53, 0.69))
    );
    weights(0) = weights(0).t;
    weights(1) = weights(1).t;
    weights(2) = weights(2).t;
    biases = Array(
      DenseVector(0.42, 0.72, 0.01, 0.3),
      DenseVector(0.21, 0.87, 0.03),
      DenseVector(0.04, 0.17)
    );
    model = Network(layers, weights, biases);
    outputs = DenseMatrix(
      (0.75, 0.98),
      (0.75, 0.28)
    );
  }
  after {}
  "Forward activations" should "be equal to what we would expect" in {
      model.forward_pass(inputs);
      assert(approximatelyEqualMatrices(model.activations(1), DenseMatrix((0.67700,0.75384,0.58817,0.70566), (0.63472,0.69292,0.54391,0.64659))))
      assert(approximatelyEqualMatrices(model.activations(2), DenseMatrix((0.87519,0.89296,0.81480), (0.86020,0.88336,0.79791))))
      assert(approximatelyEqualMatrices(model.activations(3), DenseMatrix((0.83318, 0.84132), (0.82953,0.83832))))
  }

  "Forward activations" should "be correct for a final softmax layer" in {
      model.final_activation = Utils.softmax;
      model.forward_pass(inputs);
      assert(approximatelyEqualMatrices(model.activations(3), DenseMatrix((0.4850698, 0.5149302), (0.4841319, 0.5158681))))
  }

    "Deltas and gradients" should "be correct for the backprop algorithm" in {
      model.final_activation = Utils.sigmoid;
      model.regularization = true;
      model.regularizer = 0.250;
      model.forward_pass(inputs);
      model.backprop(outputs);
      assert(model.activations.length == 4);
      assert(approximatelyEqualMatrices(model.deltas(2), DenseMatrix((0.08318,-0.13868),(0.07953,0.55832))));
      assert(approximatelyEqualMatrices(model.deltas(1), DenseMatrix((0.00639,-0.00925,-0.00779),(0.01503,0.05809,0.06892))));
      assert(approximatelyEqualMatrices(model.deltas(0), DenseMatrix((-0.00087,-0.00133,-0.00053,-0.00070), (0.01694,0.01465,0.01999,0.01622))));
      println("GRADIENT");
      println(model.gradients(0));
      assert(approximatelyEqualMatrices(model.gradients(0), DenseMatrix((0.02564, 0.01837, 0.03196, 0.05037),(0.04987, 0.06719, 0.05252, 0.08492)).t));
      assert(approximatelyEqualMatrices(model.gradients(1), DenseMatrix((0.09068, 0.06780, 0.08924), (0.02512, 0.04164, 0.12094), (0.12597, 0.05308, 0.10270), (0.11586, 0.12677, 0.03078)).t));
      assert(approximatelyEqualMatrices(model.gradients(2), DenseMatrix((0.17935, 0.19195), (0.12476, 0.30343), (0.13186, 0.25249)).t));
      assert(approximatelyEqualVectors(model.grad_biases(0), DenseVector(0.00804, 0.00666, 0.00973, 0.00776)));
  }
}