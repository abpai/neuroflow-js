import { Sequential, Layer, Value, Module, utils } from '../index.js'
import datasets from './datasets/index.js'
import weights from './weights/index.js'

const { prng, movingAverage, range } = utils

const rand = prng(1337)

const numOfInputs = 14 * 14
const latentDim = numOfInputs / 2
const initialization = 'he'
const activation = process.argv[2] || 'leakyRelu'

class Autoencoder extends Module {
  constructor(encoder, decoder) {
    super()
    this.encoder = encoder
    this.decoder = decoder
  }

  forward(inputs) {
    const encoded = this.encoder.forward(inputs)
    const encodedValues = encoded.map((v) => new Value(v.data))
    const decoded = this.decoder.forward(encodedValues)
    return decoded
  }

  parameters() {
    return [...this.encoder.parameters(), ...this.decoder.parameters()]
  }
}

const encoder = new Sequential({
  layers: [
    new Layer({
      numOfInputs,
      numOfNeurons: latentDim,
      initialization,
      activation,
      rand,
    }),
  ],
})

const decoder = new Sequential({
  layers: [
    new Layer({
      numOfInputs: latentDim,
      numOfNeurons: numOfInputs,
      initialization,
      activation,
      rand,
    }),
  ],
})

const mse = (predictions, targets) => {
  const squares = predictions.map((pred, i) => pred.sub(targets[i]).pow(2))
  return squares.reduce((a, b) => a.add(b), new Value(0)).div(targets.length)
}

const trainModel = (model, xs, steps, learningRate, alpha) => {
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
    learningRate = +(learningRate * (1 - alpha)).toFixed(6) || 0.00001

    lossHistory.push(loss.data)

    if (step % 10 === 0) {
      console.info(
        `Step: ${step}, Avg Loss: ${movingAverage(lossHistory)}, Learning Rate: ${learningRate}`,
      )
    }
  })
}

const dataset = datasets.read('mnist', 'train')
const xs = dataset.map(({ image }) => image)

const steps = 20000
const learningRate = 1
const alpha = 1e-4
const model = new Autoencoder(encoder, decoder)

trainModel(model, xs, steps, learningRate, alpha)

weights.write(`mnist-encoder-${activation}`, model.encoder.weights())
weights.write(`mnist-decoder-${activation}`, model.decoder.weights())
