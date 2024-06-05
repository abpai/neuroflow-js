import Module from './module.js'

export default class Sequential extends Module {
  constructor({ layers = [] } = {}) {
    super()
    this.layers = layers
  }

  add(layer) {
    this.layers.push(layer)
  }

  forward(inputs) {
    const output = this.layers.reduce(
      (input, layer) => layer.forward(input),
      inputs,
    )
    return output
  }

  parameters() {
    return this.layers.flatMap((layer) => layer.parameters())
  }

  weights() {
    return this.layers.map((layer) =>
      layer.neurons.map((neuron) => ({
        weights: neuron.weights.map((w) => w.data),
        bias: neuron.bias.data,
      })),
    )
  }

  toString() {
    return `Sequential of [${this.layers.map((layer) => layer.toString()).join(', ')}]`
  }
}
