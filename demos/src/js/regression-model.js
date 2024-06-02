import { Sequential, Layer } from '@andypai/neuroflow'
import range from './range.js'

const randomNumber = (min, max) => Math.random() * (max - min) + min

export const generateTrainingData = (
  xRange,
  fn = (x) => x,
  { n = 100 } = {},
) => {
  const xs = []
  const ys = []
  const { min, max } = xRange

  range(0, n).forEach(() => {
    const x = randomNumber(min, max)
    xs.push([x])
    ys.push([fn(x)])
  })

  return [xs, ys]
}

export const buildModel = () => {
  const numOfInputs = 1
  const layer1 = new Layer({ numOfInputs, numOfNeurons: 5 })
  const layer2 = new Layer({
    numOfInputs: 5,
    numOfNeurons: 10,
  })
  const layer3 = new Layer({
    numOfInputs: 10,
    numOfNeurons: 10,
  })
  const layer4 = new Layer({
    numOfInputs: 10,
    numOfNeurons: 5,
  })
  const layer5 = new Layer({
    numOfInputs: 5,
    numOfNeurons: numOfInputs,
    activation: 'linear',
  })
  const model = new Sequential({
    layers: [layer1, layer2, layer3, layer4, layer5],
  })
  return model
}
