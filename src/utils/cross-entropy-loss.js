import Value from '../engine.js'

const crossEntropyLoss = (predictedProbabilities, labels) =>
  predictedProbabilities
    .map((p, j) => new Value(-labels[j]).mul(p.log()))
    .reduce((a, b) => a.add(b), new Value(0))

export default crossEntropyLoss
