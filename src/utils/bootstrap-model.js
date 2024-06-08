import Sequential from '../sequential.js'
import Layer from '../layer.js'
import Neuron from '../neuron.js'
import Value from '../engine.js'

const bootstrapModel = (structure, linearActivation = 'softmax') => {
  const model = new Sequential({
    layers: structure.map((layer, index) => {
      const isLast = index === structure.length - 1
      return new Layer({
        neurons: layer.map(
          ({ weights, bias }) =>
            new Neuron({
              weights: weights.map((w) => new Value(w)),
              bias,
              activation: isLast ? linearActivation : 'relu',
            }),
        ),
        activation: isLast ? linearActivation : 'relu',
      })
    }),
  })
  return model
}

export default bootstrapModel
