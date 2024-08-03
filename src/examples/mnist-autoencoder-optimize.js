import Autoencoder from '../utils/autoencoder.js'
import Value from '../engine.js'
import datasets from './datasets/index.js'
import weights from './weights/index.js'
import utils from '../utils/index.js'

const { prng, movingAverage, range } = utils

const rand = prng(1337)

const activation = process.argv[2] || 'leakyRelu'
const dataset = datasets.read('mnist', 'train')
const encoderWeights = weights.read(`mnist-encoder-${activation}`)
const decoderWeights = weights.read(`mnist-decoder-${activation}`)

const encoder = utils.bootstrapModel(encoderWeights, activation)
const decoder = utils.bootstrapModel(decoderWeights, activation)
const model = new Autoencoder(encoder, decoder)

const mse = (predictions, targets) => {
  const squares = predictions.map((pred, i) => pred.sub(targets[i]).pow(2))
  return squares.reduce((a, b) => a.add(b), new Value(0)).div(targets.length)
}

const trainModel = (xs, steps, learningRate, alpha) => {
  const lossHistory = []

  range(0, steps).forEach((step) => {
    const idx = Math.floor(rand() * 10)
    const x = xs[idx]
    const pixels = model.forward(x)
    const loss = mse(pixels, x)

    model.zeroGrad()
    loss.backward()

    // Update model parameters in the opposite direction of the gradient
    model.parameters().forEach((param) => {
      param.data -= learningRate * param.grad
    })

    // Adjust learning rate
    learningRate = +(learningRate * (1 - alpha)).toFixed(7) || 0.00001

    lossHistory.push(loss.data)

    if (step % 10 === 0) {
      console.info(
        `Step: ${step}, Avg Loss: ${movingAverage(lossHistory)}, Learning Rate: ${learningRate}`,
      )
    }
  })
}

const steps = 10000
const learningRate = 0.1
const alpha = 1e-4
const xs = dataset.map(({ image }) => image)

trainModel(xs, steps, learningRate, alpha)

weights.write(`mnist-encoder-${activation}`, model.encoder.weights())
weights.write(`mnist-decoder-${activation}`, model.decoder.weights())
