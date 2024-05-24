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
    return this.layers.reduce((input, layer) => layer.forward(input), inputs)
  }

  parameters() {
    return this.layers.flatMap((layer) => layer.parameters())
  }

  toString() {
    return `Sequential of [${this.layers.map((layer) => layer.toString()).join(', ')}]`
  }
}
