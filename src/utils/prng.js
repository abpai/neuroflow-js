/* eslint-disable no-bitwise */
/*
 * Pseudo-random number generator (PRNG) based on the splitmix32 algorithm.
 * Allows for seeding the generator with a given seed making it possible to
 * reproduce the same sequence of pseudo-random numbers.
 *
 * The splitmix32 algorithm is a very fast PRNG that is suitable for most
 * applications. It is not cryptographically secure but provides a good
 * compromise between speed and randomness.
 *
 * Source: https://stackoverflow.com/a/47593316/2627079
 */

const splitmix32 = (a) => () => {
  a |= 0
  a = (a + 0x9e3779b9) | 0
  let t = a ^ (a >>> 16)
  t = Math.imul(t, 0x21f0aaad)
  t ^= t >>> 15
  t = Math.imul(t, 0x735a2d97)
  // eslint-disable-next-line no-return-assign
  return ((t ^= t >>> 15) >>> 0) / 4294967296
}

const prng = (seed) => splitmix32((seed * 2 ** 32) >>> 0)

export default prng
