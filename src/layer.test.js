import Layer from './layer.js'
import Neuron from './neuron.js'
import Value from './engine.js'

// Based on python random.seed(1)
const weights1 = [-0.73127, 0.69486, 0.52754].map((v) => new Value(v))
const weights2 = [-0.48986, -0.00912, -0.10101].map((v) => new Value(v))

const neurons = [
  new Neuron({ weights: weights1 }),
  new Neuron({ weights: weights2 }),
]

test('#forward', () => {
  const layer = new Layer({ neurons })
  const l = layer.forward([1, 2, 3])
  expect(l[0].data).toBeCloseTo(2.24111)
  expect(l[1].data).toBeCloseTo(0)
})

test('#parameters', () => {
  const layer = new Layer({ neurons })
  const params = layer.parameters()
  expect(params.length).toBe(8)
  expect(params[0].data).toBeCloseTo(weights1[0].data)
  expect(params[1].data).toBeCloseTo(weights1[1].data)
  expect(params[2].data).toBeCloseTo(weights1[2].data)
  expect(params[3].data).toBe(0)
  expect(params[4].data).toBeCloseTo(weights2[0].data)
  expect(params[5].data).toBeCloseTo(weights2[1].data)
  expect(params[6].data).toBeCloseTo(weights2[2].data)
  expect(params[7].data).toBe(0)
})

test('#toString', () => {
  const layer = new Layer({ numOfInputs: 2, numOfNeurons: 3 })
  expect(layer.toString()).toBe(
    'Layer of [RELUNeuron(2), RELUNeuron(2), RELUNeuron(2)]',
  )
})
