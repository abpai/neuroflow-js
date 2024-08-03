const movingAverage = (history, lookBack = 10) => {
  const sliced = history.slice(
    Math.max(0, history.length - lookBack),
    history.length,
  )
  const avg = sliced.reduce((acc, l) => acc + l, 0) / sliced.length
  return +avg.toFixed(6)
}

export default movingAverage
