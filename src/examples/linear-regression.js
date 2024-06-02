import { Sequential, Layer } from '../index.js'

// Specify the architecture
const layer1 = new Layer({ numOfInputs: 3, numOfNeurons: 4 })
const layer2 = new Layer({ numOfInputs: 4, numOfNeurons: 3 })
const layer3 = new Layer({
  numOfInputs: 3,
  numOfNeurons: 1,
  activation: 'linear',
})
const model = new Sequential({
  layers: [layer1, layer2, layer3],
})

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

const ys = [1.0, -1.0, -1.0, 1.0]

const range = (n) => [...Array(n).keys()]

const epochs = 15

range(epochs).forEach((epoch) => {
  // forward pass
  const yPredictions = xs.map((x) => model.forward(x))

  // mean squared error (mse)
  const loss = ys.reduce(
    (acc, yTarget, i) => yPredictions[i].sub(yTarget).pow(2).add(acc),
    0,
  )

  // backward pass
  model.zeroGrad()
  loss.backward()

  // update weights
  const learingRate = 0.01
  model.parameters().forEach((p) => {
    p.data -= learingRate * p.grad
  })

  console.info(`Epoch: ${epoch + 1}, Loss: ${loss.data}`)
})

model.forward([2.0, 3.0, -1.0]) // Output: 1
