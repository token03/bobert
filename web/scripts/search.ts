import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import catalog from '../search-catalog.json'

const output = new URL('../public/search.bin', import.meta.url)
const cached = existsSync(output) ? await readFile(output) : null

if (!cached || createHash('sha256').update(cached).digest('hex') !== catalog.sha256) {
  const response = await fetch(catalog.url)
  if (!response.ok) throw new Error(`Could not download search catalog: ${response.status}`)
  const bytes = new Uint8Array(await response.arrayBuffer())
  if (createHash('sha256').update(bytes).digest('hex') !== catalog.sha256) {
    throw new Error('Search catalog checksum mismatch')
  }
  await mkdir(new URL('../public/', import.meta.url), { recursive: true })
  await writeFile(output, bytes)
  console.log(`Downloaded search catalog (${(bytes.length / 1e6).toFixed(2)} MB)`)
}
