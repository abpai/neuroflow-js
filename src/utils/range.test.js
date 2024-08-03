import range from './range.js'

test('range with default step of one', () => {
  const values = range(0, 5)
  expect(values).toEqual([0, 1, 2, 3, 4])
})

test('range with step', () => {
  const values = range(0, 5, 2)
  expect(values).toEqual([0, 2, 4])
})
