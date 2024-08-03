import { Sequential, Layer, Value, utils } from '../index.js'
import datasets from './datasets/index.js'
import weights from './weights/index.js'

const { prng, bootstrapModel, oneHot, crossEntropyLoss, movingAverage, range } =
  utils

const uniqueLabels = 10
const rand = prng(1337)

const pct = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
}).format

// Specify the architecture
const buildModel = (bootstrap) => {
  let model
  if (bootstrap) {
    const layers = weights.read('mnist-classifier')
    model = bootstrapModel(layers)
    return model
  }

  const numOfInputs = 14 * 14
  const numOfNeurons = 20
  const initialization = 'he'
  const layer1 = new Layer({ numOfInputs, numOfNeurons, rand, initialization })
  const layer2 = new Layer({
    numOfInputs: numOfNeurons,
    numOfNeurons,
    rand,
    initialization,
  })
  const layer3 = new Layer({
    numOfInputs: numOfNeurons,
    numOfNeurons: uniqueLabels,
    rand,
    activation: 'softmax',
    initialization,
  })
  model = new Sequential({
    layers: [layer1, layer2, layer3],
  })
  return model
}

const trainModel = (
  model,
  xs,
  ys,
  epochs,
  batchSize,
  learningRate,
  alpha,
  lambda,
) => {
  let lossLastEpoch = Infinity
  const lossHistory = []
  const accuracyHistory = []

  range(0, epochs).forEach((epoch) => {
    const shuffledIndices = range(0, xs.length).sort(() => rand() - 0.5)
    const batches = range(0, Math.ceil(xs.length / batchSize)).map((batch) =>
      shuffledIndices.slice(batch * batchSize, (batch + 1) * batchSize),
    )

    batches.forEach((batchIdx, batchNumber) => {
      const batchXs = batchIdx.map((idx) => xs[idx])
      const batchYs = batchIdx.map((idx) => ys[idx])

      let loss = new Value(0)
      const predictions = []

      batchXs.forEach((x, j) => {
        const prediction = model.forward(x)
        predictions.push(prediction)
        const label = oneHot.encode(batchYs[j], uniqueLabels)
        loss = loss.add(crossEntropyLoss(prediction, label))
      })

      // L2 Regularization term
      const l2Loss = model
        .parameters()
        .reduce((acc, param) => acc.add(param.mul(param)), new Value(0))
        .mul(lambda / 2)

      // Add L2 regularization term to the loss
      loss = loss.add(l2Loss).div(batchSize)

      model.zeroGrad()
      loss.backward()

      // Update model parameters in the opposite direction of the gradient
      model.parameters().forEach((param) => {
        param.data -= learningRate * param.grad
      })

      // Adjust learning rate
      learningRate = +(learningRate * (1 - alpha)).toFixed(8) || 0.00000001

      const accuracy =
        predictions.reduce((acc, pred, j) => {
          const label = batchYs[j]
          const prediction = oneHot.decode(pred.map((v) => v.data))
          return acc + (label === prediction ? 1 : 0)
        }, 0) / batchSize

      lossHistory.push(loss.data)
      accuracyHistory.push(accuracy)

      if (batchNumber % 10 === 0) {
        console.info(
          `Epoch: ${epoch}.${batchNumber}, Avg Loss: ${movingAverage(lossHistory)}, Avg Accuracy: ${pct(movingAverage(accuracyHistory))}, Learning Rate: ${learningRate}`,
        )
      }
    })

    if (epoch && lossLastEpoch > movingAverage(lossHistory))
      weights.write('mnist-classifier', model.weights())
    lossLastEpoch = movingAverage(lossHistory)
  })
}

const dataset = datasets.read('mnist', 'train')
const [xs, ys] = dataset.reduce(
  (prev, data) => {
    const { image, label } = data
    prev[0].push(image)
    prev[1].push(label)
    return prev
  },
  [[], []],
)

const epochs = 3
const batchSize = 25
const learningRate = 1
const alpha = 1e-2
const lambda = 1e-3

const model = buildModel(false)
trainModel(model, xs, ys, epochs, batchSize, learningRate, alpha, lambda)

let loss = new Value(0)
const predictions = []

const testDataset = datasets.read('mnist', 'test')
const [xst, yst] = testDataset.reduce(
  (prev, data) => {
    const { image, label } = data
    prev[0].push(image)
    prev[1].push(label)
    return prev
  },
  [[], []],
)

xst.forEach((x, j) => {
  const prediction = model.forward(x)
  predictions.push(prediction)
  const label = oneHot.encode(yst[j], uniqueLabels)
  loss = loss.add(crossEntropyLoss(prediction, label))
})

loss = loss.div(xst.length)

const accuracy = predictions.reduce((acc, pred, j) => {
  const label = yst[j]
  const prediction = oneHot.decode(pred.map((v) => v.data))
  return acc + (label === prediction ? 1 : 0)
}, 0)

console.info(
  `Final, Loss: ${loss.data}, Accuracy: ${pct(accuracy / xst.length)}`,
)
