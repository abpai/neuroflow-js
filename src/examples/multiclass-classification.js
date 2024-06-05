import { Sequential, Layer, Value } from '../index.js'
import prng from '../utils/prng.js'

const rand = prng(1337)

// Specify the architecture
const layer1 = new Layer({ numOfInputs: 2, numOfNeurons: 16, rand })
const layer2 = new Layer({ numOfInputs: 16, numOfNeurons: 16, rand })
const layer3 = new Layer({
  numOfInputs: 16,
  numOfNeurons: 3,
  rand,
  activation: 'softmax',
})
const model = new Sequential({
  layers: [layer1, layer2, layer3],
})

const getRandomBetween = (low, high) => rand() * (high - low) + low

const generateLinearData = (
  nSamples,
  fn = (x, y) => (y > 0.8 && 1) || (y > 0.5 && 0) || 2,
) => {
  const data = []
  const labels = []
  Array.from({ length: nSamples }).forEach(() => {
    const x = getRandomBetween(0, 1)
    const y = getRandomBetween(0, 1)
    data.push([x, y])
    labels.push(fn(x, y))
  })
  return [data, labels]
}

const range = (n) => [...Array(n).keys()]

const [xs, ys] = generateLinearData(100)

let learningRate = 1
const alpha = 1e-4
const epochs = 100

const oneHotEncode = (label, numClasses) => {
  const encoding = Array(numClasses).fill(0)
  encoding[label] = 1
  return encoding
}

const crossEntropyLoss = (predictions, labels) => {
  const n = predictions.length
  return predictions
    .reduce((acc, pred, i) => {
      const label = labels[i]
      const loss = pred
        .map((p, j) => new Value(-label[j]).mul(p.log()))
        .reduce((a, b) => a.add(b), new Value(0))
      return acc.add(loss)
    }, new Value(0))
    .div(n)
}

const oneHotDecode = (values) => {
  const probs = values.map((v) => v.data)
  return probs.indexOf(Math.max(...probs))
}

range(epochs).forEach((epoch) => {
  const yPredictions = xs.map((input) => model.forward(input))
  const yTrue = ys.map((label) => oneHotEncode(label, 3))
  const dataLoss = crossEntropyLoss(yPredictions, yTrue)

  const regLoss = new Value(alpha).mul(
    model.parameters().reduce((acc, p) => acc.add(p.pow(2)), new Value(0)),
  )

  const totalLoss = dataLoss.add(regLoss)

  model.zeroGrad()
  totalLoss.backward()

  learningRate *= 0.99

  model.parameters().forEach((p) => {
    p.data -= learningRate * p.grad
  })

  const correct = yPredictions.filter(
    (pred, i) => oneHotDecode(pred) === ys[i],
  ).length
  const accuracy = correct / yPredictions.length

  console.info(
    `Epoch: ${epoch + 1}, Loss: ${totalLoss.data}, Accuracy: ${accuracy}`,
  )
})

console.info(oneHotDecode(model.forward([0, 1])).data) // Output: 1
