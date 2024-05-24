// eslint-disable-next-line import/no-extraneous-dependencies
import tf from '@tensorflow/tfjs-node'
import Value from './engine.js'

test('neg(): negates 2 to -2', () => {
  const two = new Value(2)
  expect(two.neg().data).toBe(-2)
})

test('add(): add 1 + 2 to equal 3', () => {
  const one = new Value(1)
  expect(one.add(2).data).toBe(3)
})

test('sub(): subtract 1 - 2 to equal -1', () => {
  const one = new Value(1)
  expect(one.sub(2).data).toBe(-1)
})

test('mul(): multiply 2 * 2 to equal 4', () => {
  const two = new Value(2)
  expect(two.mul(2).data).toBe(4)
})

test('div(): divide 4 / 2 to equal 2', () => {
  const four = new Value(4)
  expect(four.div(2).data).toBe(2)
})

test('pow(): 3^3 to equal 27', () => {
  const three = new Value(3)
  expect(three.pow(3).data).toBe(27)
})

test('relu()', () => {
  expect(new Value(-2).relu().data).toBe(0)
  expect(new Value(-1).relu().data).toBe(0)
  expect(new Value(0.5).relu().data).toBe(0.5)
  expect(new Value(1).relu().data).toBe(1)
  expect(new Value(2).relu().data).toBe(2)
})

test('tanh()', () => {
  expect(new Value(3).tanh().data).toBeCloseTo(Math.tanh(3))
  expect(new Value(0).tanh().data).toBeCloseTo(Math.tanh(0))
  expect(new Value(-3).tanh().data).toBeCloseTo(Math.tanh(-3))
})

test('chained expression', () => {
  const a = new Value(2)
  const b = new Value(-3)
  const e = a.mul(b)

  const c = new Value(10)
  const d = c.add(e)

  const f = new Value(-2)
  const l = d.mul(f)

  l.backward()

  expect(a.data).toBeCloseTo(2)
  expect(b.data).toBeCloseTo(-3)
  expect(e.data).toBeCloseTo(-6)
  expect(c.data).toBeCloseTo(10)
  expect(d.data).toBeCloseTo(4)
  expect(f.data).toBeCloseTo(-2)
  expect(l.data).toBeCloseTo(-8)
})

test('backward(): simple', () => {
  const a = new Value(2)
  const b = new Value(-3)
  const e = a.mul(b)

  const c = new Value(10)
  const d = c.add(e)

  const f = new Value(-2)
  const l = d.mul(f)

  l.backward()

  expect(a.grad).toBeCloseTo(6)
  expect(b.grad).toBeCloseTo(-4)
  expect(e.grad).toBeCloseTo(-2)
  expect(c.grad).toBeCloseTo(-2)
  expect(d.grad).toBeCloseTo(-2)
  expect(f.grad).toBeCloseTo(4)
  expect(l.grad).toBeCloseTo(1)
})

test('backward(): complex', () => {
  const x = new Value(-4.0)
  const z = x.mul(2).add(2).add(x)
  const q = z.mul(x).add(z.relu())
  const h = z.mul(z).relu()
  const y = h.add(q).add(q.mul(x))
  y.backward()

  expect(x.data).toBeCloseTo(-4)
  expect(y.data).toBeCloseTo(-20)
  expect(y.grad).toBeCloseTo(1)
  expect(x.grad).toBeCloseTo(46)
})

test('against tensorflow', () => {
  const x = new Value(-4.0)
  const z = x.mul(2).add(2).add(x)
  const q = z.mul(x).add(z.relu())
  const h = z.mul(z).relu()
  const y = h.add(q).add(q.mul(x))
  y.backward()

  const tensorX = tf.scalar(-4.0)
  const tensorZ = tf.add(tf.add(tf.mul(tensorX, 2), 2), tensorX)
  const tensorQ = tf.add(tf.mul(tensorZ, tensorX), tf.relu(tensorZ))
  const tensorH = tf.relu(tf.mul(tensorZ, tensorZ))
  const tensorY = tf.add(tf.add(tensorH, tensorQ), tf.mul(tensorQ, tensorX))

  const tfResult = tensorY.dataSync()[0]
  expect(y.data).toBeCloseTo(tfResult)

  const gradX = tf.grad((_x) => {
    const _tensorZ = tf.add(tf.add(tf.mul(_x, 2), 2), _x)
    const _tensorQ = tf.add(tf.mul(_tensorZ, _x), tf.relu(_tensorZ))
    const _tensorH = tf.relu(tf.mul(_tensorZ, _tensorZ))
    const _tensorY = tf.add(tf.add(_tensorH, _tensorQ), tf.mul(_tensorQ, _x))
    return _tensorY
  })

  const grad = gradX(tensorX).dataSync()[0]
  expect(x.grad).toBeCloseTo(grad)
})
