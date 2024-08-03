import fs from 'fs'
import { join } from 'path'

const __dirname = import.meta.dirname

export default {
  read: (type) =>
    JSON.parse(fs.readFileSync(join(__dirname, `${type}.json`), 'utf-8')),
  write: (type, weights) =>
    fs.writeFileSync(
      join(__dirname, `${type}.json`),
      JSON.stringify(weights),
      'utf-8',
    ),
}
