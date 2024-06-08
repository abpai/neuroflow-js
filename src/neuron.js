import Module from './module.js'
import Value from './engine.js'

export default class Neuron extends Module {
  constructor({
    numOfInputs = 1,
    activation = 'relu',
    initialization,
    weights,
    bias = 0,
    rand = Math.random,
  }) {
    super()

    const initValue = () =>
      initialization === 'he'
        ? (rand() - 0.5) * Math.sqrt(2 / numOfInputs)
        : rand() * 2 - 1

    this.weights =
      weights ||
      Array.from({ length: numOfInputs }, () => new Value(initValue()))
    this.bias = new Value(bias === undefined ? initValue() : bias)
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
