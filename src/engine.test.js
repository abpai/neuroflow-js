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

test('exp(): e^x', () => {
  const one = new Value(1)
  const e = one.exp()
  expect(e.data).toBeCloseTo(Math.exp(1))
  expect(e.grad).toBe(0)

  e.backward()
  expect(one.grad).toBeCloseTo(Math.exp(1))
  expect(e.grad).toBe(1)
})

test('log(): ln(x)', () => {
  const e = new Value(Math.exp(1))
  const log = e.log()
  expect(log.data).toBeCloseTo(1)
  expect(log.grad).toBe(0)

  log.backward()
  expect(e.grad).toBeCloseTo(1 / Math.exp(1))
  expect(log.grad).toBeCloseTo(1)
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

test('Value.softmax()', () => {
  const inputs = [1, 2, 3, 4, 1, 2, 3].map((v) => new Value(v))
  const probs = Value.softmax(inputs).map((p) => p.data)
  expect(probs[0]).toBeCloseTo(0.02364)
  expect(probs[1]).toBeCloseTo(0.06426)
  expect(probs[2]).toBeCloseTo(0.17468)
  expect(probs[3]).toBeCloseTo(0.47483)
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

test('backward(): softmax', () => {
  const logits = [1, 2, 3, 4].map((v) => new Value(v))
  const probs = Value.softmax(logits)
  const label = [0, 0, 0, 1]
  const loss = probs
    .map((p, j) => new Value(-label[j]).mul(p.log()))
    .reduce((a, b) => a.add(b), new Value(0))

  loss.backward()

  expect(loss.data).toBeCloseTo(
    -Math.log(
      Math.exp(4) / (Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(4)),
    ),
  )

  expect(logits[0].grad).toBeCloseTo(0.032058603)
  expect(logits[1].grad).toBeCloseTo(0.0871443)
  expect(logits[2].grad).toBeCloseTo(0.23688)
  expect(logits[3].grad).toBeCloseTo(-0.356085)
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
