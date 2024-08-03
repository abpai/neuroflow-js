import Value from '../engine.js'
import Module from '../module.js'

export default class Autoencoder extends Module {
  constructor(encoder, decoder) {
    super()
    this.encoder = encoder
    this.decoder = decoder
  }

  forward(inputs) {
    const encoded = this.encoder.forward(inputs)
    const encodedValues = encoded.map((v) => new Value(v.data))
    const decoded = this.decoder.forward(encodedValues)
    return decoded
  }

  parameters() {
    return [...this.encoder.parameters(), ...this.decoder.parameters()]
  }

  toString() {
    return `Sequential of [${[...this.encoder.layers, ...this.decoder.layers]
      .map((layer) => layer.toString())
      .join(', ')}]`
  }
}
