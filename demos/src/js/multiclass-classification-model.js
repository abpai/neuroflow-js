import { Sequential, Layer } from '@andypai/neuroflow'

const buildModel = () => {
  const layer1 = new Layer({ numOfInputs: 2, numOfNeurons: 16 })
  const layer2 = new Layer({ numOfInputs: 16, numOfNeurons: 16 })
  const layer3 = new Layer({
    numOfInputs: 16,
    numOfNeurons: 3,
    activation: 'softmax',
  })
  const model = new Sequential({
    layers: [layer1, layer2, layer3],
  })
  return model
}

export default buildModel
