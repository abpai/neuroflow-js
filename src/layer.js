import Module from './module.js'
import Neuron from './neuron.js'

export default class Layer extends Module {
  // Initializes a layer with a given number of inputs and outputs, and an activation function
  constructor({ numOfInputs, numOfNeurons, activation = 'relu', neurons }) {
    super()

    // Create an array of neurons
    this.neurons =
      neurons ||
      Array.from(
        { length: numOfNeurons },
        () => new Neuron({ numOfInputs, activation }),
      )
  }

  // Performs the forward pass for the layer
  forward(inputs) {
    // Forward pass through each neuron
    const outputs = this.neurons.map((neuron) => neuron.forward(inputs))
    // Return a single output if there is only one neuron, otherwise return an array of outputs
    return outputs.length === 1 ? outputs[0] : outputs
  }

  // Returns the list of parameters for all neurons in the layer
  parameters() {
    return this.neurons.flatMap((neuron) => neuron.parameters())
  }

  // Returns a string representation of the layer
  toString() {
    return `Layer of [${this.neurons.map((neuron) => neuron.toString()).join(', ')}]`
  }
}
