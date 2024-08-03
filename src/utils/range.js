const range = (start, end, step = 1) => {
  const length = Math.ceil((end - start) / step)
  return Array.from({ length }, (_, i) => start + i * step)
}

export default range
