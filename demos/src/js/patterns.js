const getRandomBetween = (low, high) => Math.random() * (high - low) + low

export const generateConcentricCirclesData = (nSamples) => {
  const data = []
  const labels = []
  Array.from({ length: nSamples / 2 }).forEach(() => {
    const r = getRandomBetween(0.05, 0.2)
    const t = Math.random() * 2 * Math.PI
    data.push([0.5 + r * Math.sin(t), 0.5 + r * Math.cos(t)])
    labels.push(1)
  })
  Array.from({ length: nSamples / 2 }).forEach(() => {
    const r = getRandomBetween(0.3, 0.45)
    const t = Math.random() * 2 * Math.PI
    data.push([0.5 + r * Math.sin(t), 0.5 + r * Math.cos(t)])
    labels.push(-1)
  })
  return [data, labels]
}

export const generateSpiralData = (nSamples) => {
  const data = []
  const labels = []
  Array.from({ length: nSamples / 2 }).forEach((_, i) => {
    const r = (i / (nSamples / 2)) * 0.5
    const t = ((1.25 * i) / (nSamples / 2)) * 2 * Math.PI
    data.push([0.5 + r * Math.sin(t), 0.5 + r * Math.cos(t)])
    labels.push(1)
  })
  Array.from({ length: nSamples / 2 }).forEach((_, i) => {
    const r = (i / (nSamples / 2)) * 0.5
    const t = ((1.25 * i) / (nSamples / 2)) * 2 * Math.PI + Math.PI
    data.push([0.5 + r * Math.sin(t), 0.5 + r * Math.cos(t)])
    labels.push(-1)
  })
  return [data, labels]
}

export const generateLinearData = (
  nSamples,
  fn = (x, y) => (y > 0.4 * x + 0.2 && -1) || 1,
) => {
  const data = []
  const labels = []
  Array.from({ length: nSamples }).forEach(() => {
    const x = getRandomBetween(0, 0.99)
    const y = getRandomBetween(0, 0.99)
    data.push([x, y])
    labels.push(fn(x, y))
  })
  return [data, labels]
}

const generatePatternData = (shape, nSamples) => {
  switch (shape) {
    case 'circles':
      return generateConcentricCirclesData(nSamples)
    case 'spirals':
      return generateSpiralData(nSamples)
    case 'linear':
      return generateLinearData(nSamples)
    default:
      return []
  }
}

export default generatePatternData
