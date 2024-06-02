export default (from, to, increment = 1) =>
  [...Array(Math.ceil((to - from) / increment)).keys()].map(
    (i) => from + i * increment,
  )
