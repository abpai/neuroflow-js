import oneHot from './one-hot.js'

describe('oneHot utility functions', () => {
  test('should correctly encode a label into one-hot format', () => {
    const numClasses = 10
    const label = 3
    const expected = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    const result = oneHot.encode(label, numClasses)
    expect(result).toEqual(expected)
  })

  test('should correctly decode a one-hot encoded array back to label', () => {
    const encoded = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    const expected = 3
    const result = oneHot.decode(encoded)
    expect(result).toBe(expected)
  })
})
