import datasets from './datasets/index.js'
import weights from './weights/index.js'
import Autoencoder from '../utils/autoencoder.js'
import utils from '../utils/index.js'

const draw = (image) => {
  const colorMap = {
    0: '\x1b[47m  \x1b[0m', // White background
    1: '\x1b[40m  \x1b[0m', // Black background
  }

  const WIDTH = 14

  for (let i = 0; i < image.length; i += WIDTH) {
    console.info(
      image
        .slice(i, i + WIDTH)
        .map((val) => colorMap[val])
        .join(''),
    )
  }
}

const activation = process.argv[2] || 'leakyRelu'
const dataset = datasets.read('mnist', 'train')
const encoderWeights = weights.read(`mnist-encoder-${activation}`)
const decoderWeights = weights.read(`mnist-decoder-${activation}`)

const encoder = utils.bootstrapModel(encoderWeights, activation)
const decoder = utils.bootstrapModel(decoderWeights, activation)
const model = new Autoencoder(encoder, decoder)

utils.range(0, 10, 1).forEach((label) => {
  console.info(`------------ Label: ${label} ------------`)
  const original = dataset.find((n) => n.label === label).image
  const decoded = model.forward(original)
  const reconstructed = decoded.map((y) => (y.data > 0.5 ? 1 : 0))
  draw(reconstructed)
  console.info(`----------------------------------------`)
})
