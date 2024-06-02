import { Sequential, Layer } from '@andypai/neuroflow'

const buildModel = () => {
  const numOfInputs = 2
  const layer1 = new Layer({ numOfInputs, numOfNeurons: 16 })
  const layer2 = new Layer({
    numOfInputs: 16,
    numOfNeurons: 16,
  })
  const layer3 = new Layer({
    numOfInputs: 16,
    numOfNeurons: 1,
    activation: 'linear',
  })
  const model = new Sequential({
    layers: [
      layer1, // 2 -> 16
      layer2, // 16 -> 16
      layer3, // 16 -> 1
    ],
  })
  return model
}

export default buildModel
