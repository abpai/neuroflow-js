import crossEntropyLoss from './cross-entropy-loss.js'
import { Value } from '../index.js'

describe('crossEntropyLoss', () => {
  it('calculates the correct loss for single prediction and label', () => {
    const predictions = [new Value(0.1), new Value(0.9)]
    const labels = [0, 1]
    const loss = crossEntropyLoss(predictions, labels)
    expect(loss.data).toBeCloseTo(-Math.log(0.9))
  })

  it('handles cases where prediction probabilities are zero', () => {
    const predictions = [new Value(1), new Value(0)]
    const labels = [0, 1]
    const loss = crossEntropyLoss(predictions, labels)
    expect(loss.data).toBeCloseTo(-Math.log(1e-8)) // epsilon
  })
})
