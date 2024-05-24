import Neuron from './neuron.js'
import Value from './engine.js'

// Based on python random.seed(1)
const weights = [-0.73127, 0.69486, 0.52754].map((v) => new Value(v))

test('#forward', () => {
  const neuron = new Neuron({ weights })
  const n = neuron.forward([1, 2, 3])
  expect(n.data).toBeCloseTo(2.24111)
})

test('#parameters', () => {
  const neuron = new Neuron({ weights })
  const params = neuron.parameters()
  expect(params.length).toBe(4)
  expect(params[0].data).toBeCloseTo(weights[0].data)
  expect(params[1].data).toBeCloseTo(weights[1].data)
  expect(params[2].data).toBeCloseTo(weights[2].data)
  expect(params[3].data).toBe(0)
})

test('#toString', () => {
  const neuron = new Neuron({
    numOfInputs: 3,
    weights,
  })
  expect(neuron.toString()).toBe('RELUNeuron(3)')
})
