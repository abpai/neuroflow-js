import movingAverage from './moving-average.js'

describe('movingAverage', () => {
  test('calculates the moving average of the last 10 elements by default', () => {
    const history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    const result = movingAverage(history)
    expect(result).toBe(10.5)
  })

  test('calculates the moving average with a custom lookBack period', () => {
    const history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    const result = movingAverage(history, 5)
    expect(result).toBe(13)
  })

  test('returns NaN if the history is empty', () => {
    const history = []
    const result = movingAverage(history)
    expect(result).toBe(NaN)
  })

  test('handles history shorter than the lookBack period', () => {
    const history = [1, 2, 3]
    const result = movingAverage(history, 5)
    expect(result).toBe(2)
  })

  test('handles history with negative numbers', () => {
    const history = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    const result = movingAverage(history)
    expect(result).toBe(-5.5)
  })

  test('handles history with floating point numbers', () => {
    const history = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]
    const result = movingAverage(history)
    expect(result).toBe(5.96)
  })
})
