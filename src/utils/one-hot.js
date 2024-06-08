const decode = (probs) => probs.indexOf(Math.max(...probs))

const encode = (label, numClasses) => {
  const encoding = Array(numClasses).fill(0)
  encoding[label] = 1
  return encoding
}

export default {
  decode,
  encode,
}
