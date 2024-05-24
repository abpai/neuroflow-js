import Sequential from './sequential.js'
import Layer from './layer.js'
import Neuron from './neuron.js'
import Value from './engine.js'

// Based on python random.seed(1)
const weights1 = [-0.73127, 0.69486, 0.52754].map((v) => new Value(v))
const weights2 = [-0.48986, -0.00912, -0.10101].map((v) => new Value(v))
const weights3 = [0.30318, 0.57744, -0.81228].map((v) => new Value(v))
const weights4 = [-0.9433, 0.67153, -0.13446].map((v) => new Value(v))
const weights5 = [0.52456, -0.99578, -0.10922].map((v) => new Value(v))
const weights6 = [0.44308, -0.54247, 0.89054].map((v) => new Value(v))
const weights7 = [0.80285, -0.93882, -0.9491].map((v) => new Value(v))

const buildModel = () => {
  const layer1 = new Layer({
    neurons: [
      new Neuron({ weights: weights1 }),
      new Neuron({ weights: weights2 }),
      new Neuron({ weights: weights3 }),
    ],
  })
  const layer2 = new Layer({
    neurons: [
      new Neuron({ weights: weights4 }),
      new Neuron({ weights: weights5 }),
      new Neuron({ weights: weights6 }),
    ],
  })
  const layer3 = new Layer({
    neurons: [new Neuron({ weights: weights7, activation: 'linear' })],
  })
  const sequential = new Sequential({ layers: [layer1, layer2, layer3] })
  return sequential
}

test('#forward', () => {
  const model = buildModel()
  expect(model.forward([2.0, 3.0, -1.0]).data).toBeCloseTo(-2.70302)
})

test('#parameters', () => {
  const model = new Sequential()
  model.add(new Layer({ numOfInputs: 2, numOfNeurons: 3 }))
  model.add(new Layer({ numOfInputs: 3, numOfNeurons: 2 }))
  model.add(new Layer({ numOfInputs: 2, numOfNeurons: 1 }))
  const params = model.parameters()
  // eslint-disable-next-line prettier/prettier
  expect(params.length).toBe(
    (2 * 3 + 3) +
    (3 * 2 + 2) +
    (2 * 1 + 1),
  )
})

test('train()', () => {
  const range = (n) => [...Array(n).keys()]
  const model = buildModel()

  const xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]

  const ys = [1.0, -1.0, -1.0, 1.0]

  range(200).forEach(() => {
    // forward pass
    const yPredictions = xs.map((x) => model.forward(x))
    const loss = ys.reduce(
      (acc, yTarget, i) => yPredictions[i].sub(yTarget).pow(2).add(acc),
      0,
    )

    // backward pass
    model.zeroGrad()
    loss.backward()

    // update weights
    const learingRate = 0.025
    model.parameters().forEach((p) => {
      p.data -= learingRate * p.grad
    })
  })

  const yPredictions = xs.map((x) => model.forward(x))
  expect(yPredictions[0].data).toBeCloseTo(1.0)
  expect(yPredictions[1].data).toBeCloseTo(-1.0)
  expect(yPredictions[2].data).toBeCloseTo(-1.0)
  expect(yPredictions[3].data).toBeCloseTo(1.0)
})

test('#toString', () => {
  const layer = new Layer({ numOfInputs: 2, numOfNeurons: 3 })
  expect(layer.toString()).toBe(
    'Layer of [RELUNeuron(2), RELUNeuron(2), RELUNeuron(2)]',
  )
})
