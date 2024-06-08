import fs from 'fs'
import { join } from 'path'
import { Sequential, Layer, Value, utils } from '../index.js'

const { prng, bootstrapModel, oneHot, crossEntropyLoss } = utils

const uniqueLabels = 10
const rand = Math.random || prng(1337)

const range = (start, end) => {
  const length = end - start
  return Array.from({ length }, (_, i) => start + i)
}

const readWeights = () =>
  fs.readFileSync(
    join(import.meta.dirname, 'datasets', 'mnist', 'weights.json'),
    'utf-8',
  )

const writeWeights = (weights) => {
  console.info('Saving weights...')
  fs.writeFileSync(
    join(import.meta.dirname, 'datasets', 'mnist', 'weights.json'),
    JSON.stringify(weights),
    'utf-8',
  )
}

const readTrainingDataset = () =>
  fs.readFileSync(
    join(import.meta.dirname, 'datasets', 'mnist', 'train.json'),
    'utf-8',
  )

const readTestDataset = () =>
  fs.readFileSync(
    join(import.meta.dirname, 'datasets', 'mnist', 'test.json'),
    'utf-8',
  )

const pct = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

// Specify the architecture
const buildModel = (bootstrap) => {
  let model
  if (bootstrap) {
    const layers = JSON.parse(readWeights())
    model = bootstrapModel(layers)
    return model
  }

  const layer1 = new Layer({ numOfInputs: 14 * 14, numOfNeurons: 20, rand })
  const layer2 = new Layer({
    numOfInputs: 20,
    numOfNeurons: 20,
    rand,
  })
  const layer3 = new Layer({
    numOfInputs: 20,
    numOfNeurons: uniqueLabels,
    rand,
    activation: 'softmax',
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

  range(0, epochs).forEach((epoch) => {
    let loss
    const shuffledIndices = range(0, xs.length).sort(() => rand() - 0.5)
    const batches = range(0, Math.ceil(xs.length / batchSize)).map((batch) =>
      shuffledIndices.slice(batch * batchSize, (batch + 1) * batchSize),
    )

    batches.forEach((batchIdx, batchNumber) => {
      const batchXs = batchIdx.map((idx) => xs[idx])
      const batchYs = batchIdx.map((idx) => ys[idx])

      loss = new Value(0)
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
      loss.backward()

      // Update model parameters in the opposite direction of the gradient
      model.parameters().forEach((param) => {
        param.data -= learningRate * param.grad
        param.grad = 0
      })

      // Adjust learning rate
      learningRate = +(learningRate * (1 - alpha)).toFixed(10) || 0.00000001

      if (batchNumber % 10 === 0) {
        const accuracy = predictions.reduce((acc, pred, j) => {
          const label = batchYs[j]
          const prediction = oneHot.decode(pred.map((v) => v.data))
          return acc + (label === prediction ? 1 : 0)
        }, 0)

        console.info(
          `Epoch: ${epoch + 1}.${batchNumber + 1}, Loss: ${loss.data}, Accuracy: ${pct.format(accuracy / batchSize)}, Learning Rate: ${learningRate}`,
        )
      }
    })

    if (epoch && lossLastEpoch > loss.data) writeWeights(model.weights())
    lossLastEpoch = loss.data
  })
}

const dataset = JSON.parse(readTrainingDataset())
const [xs, ys] = dataset.reduce(
  (prev, data) => {
    const { image, label } = data
    prev[0].push(image)
    prev[1].push(label)
    return prev
  },
  [[], []],
)

const learningRate = 0.01
const alpha = 1e-3
const epochs = 5
const batchSize = 10
const lambda = 0.01

const model = buildModel(true)
trainModel(model, xs, ys, epochs, batchSize, learningRate, alpha, lambda)

writeWeights(model.weights())

let loss = new Value(0)
const predictions = []

const testDataset = JSON.parse(readTestDataset())
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
  `Final, Loss: ${loss.data}, Accuracy: ${pct.format(accuracy / xst.length)}`,
)
