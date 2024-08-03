import { utils } from '@andypai/neuroflow'
import classifierWeights from './weights/mnist-classifier.json'

export default async () => {
  const classifier = await fetch(classifierWeights).then((res) => res.json())
  return utils.bootstrapModel(classifier)
}
