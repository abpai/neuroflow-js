import fs from 'fs'

const locals = JSON.parse(fs.readFileSync('./locals.json', 'utf8'))

export default {
  locals
}
