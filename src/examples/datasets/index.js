import fs from 'fs'
import { join } from 'path'

export default {
  read: (dataset, type) =>
    JSON.parse(
      fs.readFileSync(
        join(import.meta.dirname, dataset, `${type}.json`),
        'utf-8',
      ),
    ),
}
