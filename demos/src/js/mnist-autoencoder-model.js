import { utils } from '@andypai/neuroflow'
import leakyReluEncoderWeights from './weights/mnist-encoder-leakyRelu.json'
import leakyReluDecoderWeights from './weights/mnist-decoder-leakyRelu.json'
import reluEncoderWeights from './weights/mnist-encoder-relu.json'
import reluDecoderWeights from './weights/mnist-decoder-relu.json'
import sigmoidEncoderWeights from './weights/mnist-encoder-sigmoid.json'
import sigmoidDecoderWeights from './weights/mnist-decoder-sigmoid.json'

const { Autoencoder, bootstrapModel } = utils

export default async (activation) => {
  const encoderPath =
    (activation === 'leakyRelu' && leakyReluEncoderWeights) ||
    (activation === 'relu' && reluEncoderWeights) ||
    sigmoidEncoderWeights
  const decoderPath =
    (activation === 'leakyRelu' && leakyReluDecoderWeights) ||
    (activation === 'relu' && reluDecoderWeights) ||
    sigmoidDecoderWeights

  const [encoderWeights, decoderWeights] = await Promise.all([
    fetch(encoderPath).then((res) => res.json()),
    fetch(decoderPath).then((res) => res.json()),
  ])

  const encoder = bootstrapModel(encoderWeights, activation)
  const decoder = bootstrapModel(decoderWeights, activation)
  return new Autoencoder(encoder, decoder)
}
