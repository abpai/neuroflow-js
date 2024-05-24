import Module from './module.js'

test('#zeroGrad', () => {
  const module = new Module()
  const param1 = { grad: 1 }
  const param2 = { grad: 2 }
  module.parameters = () => [param1, param2]
  module.zeroGrad()
  expect(param1.grad).toBe(0)
  expect(param2.grad).toBe(0)
})

test('#parameters', () => {
  const module = new Module()
  expect(module.parameters()).toStrictEqual([])
})
