import Module from './module.js'
import Value from './engine.js'

export default class Neuron extends Module {
  // Initializes a neuron with a given number of inputs and an activation function
  constructor({
    numOfInputs,
    activation = 'relu',
    weights,
    rand = Math.random,
  }) {
    super()
    this.weights =
      weights ||
      // Randomly initialize weights between -1 and 1
      Array.from({ length: numOfInputs }, () => new Value(rand() * 2 - 1))

    this.bias = new Value(0)
    this.activation = activation
  }

  // Performs the forward pass for the neuron
  forward(inputs) {
    // Compute the weighted sum of inputs and bias
    const activation = this.weights.reduce(
      (sum, weight, i) => sum.add(weight.mul(inputs[i])),
      this.bias,
    )

    if (this.activation === 'relu') return activation.relu()
    if (this.activation === 'tanh') return activation.tanh()
    if (['linear', 'softmax'].includes(this.activation)) return activation
    throw new Error(`Unsupported activation function: ${this.activation}`)
  }

  // Returns the list of parameters (weights and bias)
  parameters() {
    return [...this.weights, this.bias]
  }

  // Returns a string representation of the neuron
  toString() {
    return `${this.activation.toUpperCase()}Neuron(${this.weights.length})`
  }
}
