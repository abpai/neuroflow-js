import prng from './prng.js'

test('prng', () => {
  const seed = 42
  const generator = prng(seed)
  const values = Array.from({ length: 5 }, generator)
  expect(values[0]).toBeCloseTo(0.3921251)
  expect(values[1]).toBeCloseTo(0.85059318)
  expect(values[2]).toBeCloseTo(0.68442047)
  expect(values[3]).toBeCloseTo(0.4986653)
  expect(values[4]).toBeCloseTo(0.7671982)
})
