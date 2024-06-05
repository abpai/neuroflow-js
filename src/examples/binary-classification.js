import { Sequential, Layer, Value } from '../index.js'
import prng from '../utils/prng.js'

const rand = prng(1337)

// Specify the architecture
const layer1 = new Layer({ numOfInputs: 2, numOfNeurons: 4, rand })
const layer2 = new Layer({ numOfInputs: 4, numOfNeurons: 3, rand })
const layer3 = new Layer({
  numOfInputs: 3,
  numOfNeurons: 1,
  rand,
  activation: 'linear',
})
const model = new Sequential({
  layers: [layer1, layer2, layer3],
})

const xs = [
  [1.0, 2.0],
  [2.0, 3.0],
  [3.0, 1.0],
  [4.0, 2.0],
]

const ys = [1.0, 1, -1.0, -1.0] // y > x => 1

const range = (n) => [...Array(n).keys()]

let learningRate = 0.5
const alpha = 1e-4
const epochs = 15

range(epochs).forEach((epoch) => {
  const yPredictions = xs.map((input) => model.forward(input))
  const losses = yPredictions.reduce((acc, yPrediction, index) => {
    const loss = yPrediction.mul(-ys[index]).add(1).relu()
    return loss.add(acc)
  }, new Value(0))

  const dataLoss = losses.div(yPredictions.length)

  const regLoss = new Value(alpha).mul(
    model.parameters().reduce((acc, p) => acc.add(p.pow(2)), new Value(0)),
  )

  const totalLoss = dataLoss.add(regLoss)

  const correct = yPredictions.filter(
    (y, index) => y.data > 0 === ys[index] > 0,
  ).length
  const accuracy = correct / yPredictions.length

  model.zeroGrad()
  totalLoss.backward()

  learningRate = 0.5 - (0.45 * epoch) / epochs

  model.parameters().forEach((p) => {
    p.data -= learningRate * p.grad
  })

  console.info(
    `Epoch: ${epoch + 1}, Loss: ${totalLoss.data}, Accuracy: ${accuracy}`,
  )
})

console.info(model.forward([1.0, 2.0]).data) // Output: 1
