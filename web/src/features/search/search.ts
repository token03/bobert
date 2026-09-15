const SEARCH_TERM = /[0-9a-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+/g
const COMBINING_MARKS = /[\u0300-\u036f]/g
const REPEATED_LETTER = /([0-9a-z])\1+/g

export type SearchDiff = {
  id: number
  version: string
  stars: number | null
  ar: number
  cs: number
  od: number
  hp: number
  bpm: number
  length: number
}
export type SearchSet = {
  id: number
  artist: string
  title: string
  creator: string
  status: number
  year: number | null
  diffs: SearchDiff[]
}

export type SearchCorpus = {
  setIds: Uint32Array
  setArtist: Uint32Array
  setTitle: Uint32Array
  setCreator: Uint32Array
  setStatus: Uint8Array
  setYear: Uint16Array
  setDiffStart: Uint32Array
  setDiffCount: Uint32Array
  diffIds: Int32Array
  diffVersions: Uint32Array
  diffStars: Uint32Array
  diffAr: Uint32Array
  diffCs: Uint32Array
  diffOd: Uint32Array
  diffHp: Uint32Array
  diffBpm: Uint32Array
  diffLength: Uint32Array
  mapIds: Int32Array
  mapSetIds: Uint32Array
  strings: string[]
  terms: string[]
  postingStarts: Uint32Array
  postingCounts: Uint32Array
  postingBlob: Uint8Array
}

export function tokenize(text: string): string[] {
  return foldText(text).match(SEARCH_TERM) ?? []
}

export function decodeVarint(blob: Uint8Array, cursor: { offset: number }): number {
  let value = 0
  let shift = 0
  while (true) {
    const byte = blob[cursor.offset++]
    value |= (byte & 0x7f) << shift
    if (byte < 0x80) {
      return value >>> 0
    }
    shift += 7
  }
}

export function postingDocuments(corpus: SearchCorpus, term: number): number[] {
  const cursor = { offset: corpus.postingStarts[term] }
  const documents: number[] = []
  let previous = 0
  for (let index = 0; index < corpus.postingCounts[term]; index++) {
    previous += decodeVarint(corpus.postingBlob, cursor)
    documents.push(previous)
  }
  return documents
}

export function setDiffs(corpus: SearchCorpus, set: number): SearchDiff[] {
  const start = corpus.setDiffStart[set]
  const diffs: SearchDiff[] = []
  for (let index = 0; index < corpus.setDiffCount[set]; index++) {
    const diff = start + index
    diffs.push({
      id: corpus.diffIds[diff],
      version: corpus.strings[corpus.diffVersions[diff]],
      stars: corpus.diffStars[diff] / 100,
      ar: corpus.diffAr[diff] / 10,
      cs: corpus.diffCs[diff] / 10,
      od: corpus.diffOd[diff] / 10,
      hp: corpus.diffHp[diff] / 10,
      bpm: corpus.diffBpm[diff] / 10,
      length: corpus.diffLength[diff],
    })
  }
  return diffs.sort((left, right) => (right.stars ?? 0) - (left.stars ?? 0))
}

const RESULT_LIMIT = 30
const RERANK_LIMIT = 300
const FUZZY_RERANK_LIMIT = 60
const TEXT_SCAN_LIMIT = 5000
const STRICT_MINIMUM = 10
const PREFIX_LIMIT = 64
const CORRECTION_LIMIT = 32
const COMPOUND_LIMIT = 6
const COMPOUND_MIN_LENGTH = 8
const COMPOUND_MIN_HALF = 4
const MAX_TYPO_DISTANCE = 2
const MAX_PREFIX_SLACK = 4
const DISTANCE_LIMIT = 32
const PREFIX_WEIGHT = 0.75
const TYPO_WEIGHTS = [0, 0.5, 0.32]
const COMPOUND_WEIGHT = 0.65
const STATUS_BOOST = 0.5
const TEXT_BOOST = 1
const TITLE_BOOST = 3
const ARTIST_BOOST = 2
const CREATOR_BOOST = 1
const FIELD_PREFIX_HIT = 0.7
const FIELD_PACKED_HIT = 0.6
const FIELD_TYPO_HITS = [0, 0.55, 0.35]
const SIMILARITY_EXACT = 4
const SIMILARITY_PREFIX = 2.5
const SIMILARITY_INCLUDES = 1.5
const POPULARITY_WEIGHT = 0.05
const POPULARITY_STARS = 15
const LATIN_TOKEN = /^[a-z0-9]+$/
const ASCII_ONLY = /^[\x20-\x7e]*$/
const DISTANCE_ROWS = [
  new Uint8Array(DISTANCE_LIMIT + 2),
  new Uint8Array(DISTANCE_LIMIT + 2),
  new Uint8Array(DISTANCE_LIMIT + 2),
]

type SearchIndex = { idf: Float32Array; masks: Uint32Array; lengths: Uint16Array }
type TokenText = { text: string; low: number; high: number }
type FieldInfo = { tokens: string[]; packed: string }
type Ranked = { document: number; score: number; stars: number }

const INDEXES = new WeakMap<SearchCorpus, SearchIndex>()
const CORRECTIONS = new Map<string, { term: number; distance: number }[]>()
const CORRECTION_CACHE_LIMIT = 512

function maskLow(text: string): number {
  let mask = 0
  for (let offset = 0; offset < text.length; offset++) {
    const code = text.charCodeAt(offset)
    if (code >= 97 && code <= 122) mask |= 1 << (code - 97)
  }
  return mask
}

function maskHigh(text: string): number {
  let mask = 0
  for (let offset = 0; offset < text.length; offset++) {
    const code = text.charCodeAt(offset)
    if (code >= 48 && code <= 57) mask |= 1 << (code - 48)
  }
  return mask
}

function buildIndex(data: SearchCorpus): SearchIndex {
  const count = data.terms.length
  const index: SearchIndex = {
    idf: new Float32Array(count),
    masks: new Uint32Array(count * 2),
    lengths: new Uint16Array(count),
  }
  const total = Math.max(data.setIds.length, 1)
  for (let term = 0; term < count; term++) {
    const text = data.terms[term]
    index.idf[term] = Math.log(1 + total / data.postingCounts[term])
    index.lengths[term] = text.length
    index.masks[term * 2] = maskLow(text)
    index.masks[term * 2 + 1] = maskHigh(text)
  }
  return index
}

function searchIndex(data: SearchCorpus): SearchIndex {
  const cached = INDEXES.get(data)
  if (cached) return cached
  const index = buildIndex(data)
  INDEXES.set(data, index)
  return index
}

function lowerBound(terms: string[], term: string): number {
  let low = 0
  let high = terms.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (terms[middle] < term) low = middle + 1
    else high = middle
  }
  return low
}

function popcount(value: number): number {
  value -= (value >>> 1) & 0x55555555
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333)
  value = (value + (value >>> 4)) & 0x0f0f0f0f
  return (value * 0x01010101) >>> 24
}

function editDistance(left: string, right: string, max: number): number {
  const leftLength = left.length
  const rightLength = right.length
  if (Math.abs(leftLength - rightLength) > max) return max + 1
  if (leftLength > DISTANCE_LIMIT || rightLength > DISTANCE_LIMIT) return max + 1
  let twoBack = DISTANCE_ROWS[0]
  let previous = DISTANCE_ROWS[1]
  let current = DISTANCE_ROWS[2]
  for (let column = 0; column <= rightLength; column++) {
    twoBack[column] = column
    previous[column] = column
  }
  for (let row = 1; row <= leftLength; row++) {
    current[0] = row
    let rowMin = row
    const leftCode = left.charCodeAt(row - 1)
    for (let column = 1; column <= rightLength; column++) {
      const rightCode = right.charCodeAt(column - 1)
      let value = Math.min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (leftCode === rightCode ? 0 : 1))
      if (row > 1 && column > 1 && leftCode === right.charCodeAt(column - 2) && left.charCodeAt(row - 2) === rightCode) {
        value = Math.min(value, twoBack[column - 2] + 1)
      }
      current[column] = value
      if (value < rowMin) rowMin = value
    }
    if (rowMin > max) return max + 1
    const recycled = twoBack
    twoBack = previous
    previous = current
    current = recycled
  }
  return previous[rightLength]
}

function correctionMatches(data: SearchCorpus, index: SearchIndex, token: TokenText, last: boolean): { term: number; distance: number }[] {
  if (token.text.length < 4 || token.text.length > DISTANCE_LIMIT || !LATIN_TOKEN.test(token.text)) return []
  const tokenLength = token.text.length
  const matches: { term: number; distance: number }[] = []
  for (let term = 0; term < data.terms.length; term++) {
    const termLength = index.lengths[term]
    if (termLength === 0) continue
    let distance = MAX_TYPO_DISTANCE + 1
    if (Math.abs(termLength - tokenLength) <= MAX_TYPO_DISTANCE) {
      if (popcount(index.masks[term * 2] ^ token.low) + popcount(index.masks[term * 2 + 1] ^ token.high) <= MAX_TYPO_DISTANCE) {
        distance = editDistance(token.text, data.terms[term], MAX_TYPO_DISTANCE)
      }
    }
    if (last && termLength > tokenLength && termLength - tokenLength <= MAX_PREFIX_SLACK) {
      const text = data.terms[term]
      if (popcount(maskLow(text.slice(0, tokenLength)) ^ token.low) + popcount(maskHigh(text.slice(0, tokenLength)) ^ token.high) <= MAX_TYPO_DISTANCE) {
        const sliced = editDistance(token.text, text.slice(0, tokenLength), MAX_TYPO_DISTANCE)
        if (sliced < distance) distance = sliced
      }
    }
    if (distance > 0 && distance <= MAX_TYPO_DISTANCE) matches.push({ term, distance })
  }
  matches.sort((left, right) =>
    left.distance - right.distance ||
    Math.abs(index.lengths[left.term] - tokenLength) - Math.abs(index.lengths[right.term] - tokenLength) ||
    data.postingCounts[right.term] - data.postingCounts[left.term])
  return matches.slice(0, CORRECTION_LIMIT)
}

function simpleMatches(data: SearchCorpus, index: SearchIndex, token: string): Map<number, number> {
  const matches = new Map<number, number>()
  const start = lowerBound(data.terms, token)
  if (data.terms[start] === token) matches.set(start, index.idf[start])
  for (let term = start; term < data.terms.length && matches.size < PREFIX_LIMIT; term++) {
    if (!data.terms[term].startsWith(token)) break
    if (!matches.has(term)) matches.set(term, PREFIX_WEIGHT)
  }
  return matches
}

function documentHits(data: SearchCorpus, terms: Map<number, number>): Map<number, number> {
  const hits = new Map<number, number>()
  for (const [term, score] of terms) {
    for (const document of postingDocuments(data, term)) {
      const existing = hits.get(document)
      if (existing === undefined || score > existing) hits.set(document, score)
    }
  }
  return hits
}

function compoundMatches(data: SearchCorpus, index: SearchIndex, token: string): Map<number, number> {
  const hits = new Map<number, number>()
  if (token.length < COMPOUND_MIN_LENGTH || !LATIN_TOKEN.test(token)) return hits
  let splits = 0
  for (let point = COMPOUND_MIN_HALF; point <= token.length - COMPOUND_MIN_HALF && splits < COMPOUND_LIMIT; point++) {
    const leftText = tokenize(token.slice(0, point))
    const rightText = tokenize(token.slice(point))
    if (leftText.length !== 1 || rightText.length !== 1) continue
    if (leftText[0].length < 3 || rightText[0].length < 3) continue
    const left = simpleMatches(data, index, leftText[0])
    if (!left.size) continue
    const right = simpleMatches(data, index, rightText[0])
    if (!right.size) continue
    splits++
    const rightHits = documentHits(data, right)
    for (const [document, value] of documentHits(data, left)) {
      const other = rightHits.get(document)
      if (other === undefined) continue
      const score = COMPOUND_WEIGHT * (value + other)
      const existing = hits.get(document)
      if (existing === undefined || score > existing) hits.set(document, score)
    }
  }
  return hits
}

function tokenHits(data: SearchCorpus, index: SearchIndex, token: TokenText, last: boolean, fuzzy: boolean): Map<number, number> {
  const hits = documentHits(data, simpleMatches(data, index, token.text))
  if (!fuzzy || hits.size >= STRICT_MINIMUM) return hits
  const key = `${last ? 'p' : 'f'}:${token.text}`
  let corrections = CORRECTIONS.get(key)
  if (!corrections) {
    corrections = correctionMatches(data, index, token, last)
    if (CORRECTIONS.size >= CORRECTION_CACHE_LIMIT) CORRECTIONS.clear()
    CORRECTIONS.set(key, corrections)
  }
  for (const { term, distance } of corrections) {
    const score = TYPO_WEIGHTS[distance] * Math.log1p(data.postingCounts[term])
    for (const document of postingDocuments(data, term)) {
      const existing = hits.get(document)
      if (existing === undefined || score > existing) hits.set(document, score)
    }
  }
  for (const [document, score] of compoundMatches(data, index, token.text)) {
    const existing = hits.get(document)
    if (existing === undefined || score > existing) hits.set(document, score)
  }
  return hits
}

function intersection(groups: Map<number, number>[]): number[] {
  if (groups.some((group) => group.size === 0)) return []
  const order = [...groups].sort((left, right) => left.size - right.size)
  const candidates: number[] = []
  for (const document of order[0].keys()) {
    let present = true
    for (let group = 1; group < order.length; group++) {
      if (!order[group].has(document)) {
        present = false
        break
      }
    }
    if (present) candidates.push(document)
  }
  return candidates
}

function fieldInfo(text: string): FieldInfo {
  const tokens = tokenize(text)
  return { tokens, packed: tokens.join('') }
}

function strongMatch(token: TokenText, field: FieldInfo): number {
  if (field.tokens.includes(token.text)) return 1
  if (token.text.length < 2) return 0
  for (const term of field.tokens) {
    if (term.startsWith(token.text)) return FIELD_PREFIX_HIT
  }
  if (token.text.length >= 5 && field.packed.includes(token.text)) return FIELD_PACKED_HIT
  return 0
}

function fuzzyMatch(token: TokenText, field: FieldInfo): number {
  if (token.text.length < 3 || !LATIN_TOKEN.test(token.text)) return 0
  let best = 0
  for (const term of field.tokens) {
    let distance = MAX_TYPO_DISTANCE + 1
    if (Math.abs(term.length - token.text.length) <= MAX_TYPO_DISTANCE) {
      if (popcount(maskLow(term) ^ token.low) + popcount(maskHigh(term) ^ token.high) <= MAX_TYPO_DISTANCE) {
        distance = editDistance(token.text, term, MAX_TYPO_DISTANCE)
      }
    }
    if (term.length > token.text.length && term.length - token.text.length <= MAX_PREFIX_SLACK && term.charCodeAt(0) === token.text.charCodeAt(0)) {
      const sliced = editDistance(token.text, term.slice(0, token.text.length), MAX_TYPO_DISTANCE)
      if (sliced < distance) distance = sliced
    }
    if (distance > 0 && distance <= MAX_TYPO_DISTANCE) {
      best = Math.max(best, FIELD_TYPO_HITS[distance])
      if (best === FIELD_TYPO_HITS[1]) return best
    }
  }
  return best
}

function fuzzyFieldBonus(tokens: TokenText[], title: FieldInfo, artist: FieldInfo, creator: FieldInfo): number {
  let fields = 0
  for (const token of tokens) {
    if (strongMatch(token, title) || strongMatch(token, artist) || strongMatch(token, creator)) continue
    fields += TITLE_BOOST * fuzzyMatch(token, title) + ARTIST_BOOST * fuzzyMatch(token, artist) + CREATOR_BOOST * fuzzyMatch(token, creator)
  }
  return fields / tokens.length
}

function foldText(text: string): string {
  const lowered = text.toLowerCase()
  const folded = ASCII_ONLY.test(lowered) ? lowered : lowered.normalize('NFKD').replace(COMBINING_MARKS, '')
  return folded.replace(REPEATED_LETTER, '$1').replace(/\s+/g, ' ').trim()
}

function similarity(needle: string, packed: string): number {
  if (!needle) return 0
  if (packed === needle) return SIMILARITY_EXACT
  if (packed.startsWith(needle)) return SIMILARITY_PREFIX
  return packed.includes(needle) ? SIMILARITY_INCLUDES : 0
}

function topStars(data: SearchCorpus, document: number): number {
  const start = data.setDiffStart[document]
  let stars = 0
  for (let index = 0; index < data.setDiffCount[document]; index++) {
    const value = data.diffStars[start + index]
    if (value > stars) stars = value
  }
  return stars
}

export function searchCorpus(data: SearchCorpus, query: string): SearchSet[] {
  const words = tokenize(query).slice(0, 12)
  if (!words.length) return []
  const index = searchIndex(data)
  const tokens: TokenText[] = words.map((text) => ({ text, low: maskLow(text), high: maskHigh(text) }))
  let groups = tokens.map((token, position) => tokenHits(data, index, token, position === tokens.length - 1, false))
  let candidates = intersection(groups)
  if (candidates.length < STRICT_MINIMUM) {
    let remaining = tokens
      .map((token, position) => tokenHits(data, index, token, position === tokens.length - 1, true))
      .filter((group) => group.size > 0)
    if (remaining.length * 2 >= tokens.length) {
      let fuzzyCandidates = intersection(remaining)
      while (!fuzzyCandidates.length && remaining.length > 1 && remaining.length * 2 >= tokens.length) {
        let bestIndex = -1
        let bestCandidates: number[] = []
        for (let index = 0; index < remaining.length; index++) {
          const found = intersection(remaining.filter((_, position) => position !== index))
          if (found.length > bestCandidates.length) {
            bestCandidates = found
            bestIndex = index
          }
        }
        if (bestIndex < 0) break
        remaining = remaining.filter((_, position) => position !== bestIndex)
        fuzzyCandidates = bestCandidates
      }
      if (fuzzyCandidates.length > candidates.length) {
        groups = remaining
        candidates = fuzzyCandidates
      }
    }
  }
  if (!candidates.length) return []
  const scanText = candidates.length <= TEXT_SCAN_LIMIT && words.some((word) => word.length >= 4)
  let ranked: Ranked[] = candidates.map((document) => {
    let score = data.setStatus[document] > 0 ? STATUS_BOOST : 0
    for (const group of groups) score += group.get(document) ?? 0
    if (scanText) {
      const folded = foldText(`${data.strings[data.setArtist[document]]} ${data.strings[data.setTitle[document]]} ${data.strings[data.setCreator[document]]}`)
      let hits = 0
      for (const token of tokens) if (folded.includes(token.text)) hits++
      score += hits * TEXT_BOOST
    }
    return { document, score, stars: topStars(data, document) }
  })
  if (ranked.length > RERANK_LIMIT) {
    ranked.sort((left, right) => right.score - left.score || right.stars - left.stars)
    ranked = ranked.slice(0, RERANK_LIMIT)
  }
  const spaced = words.join(' ')
  const packedNeedle = words.join('')
  for (const entry of ranked) {
    const title = fieldInfo(data.strings[data.setTitle[entry.document]])
    const artist = fieldInfo(data.strings[data.setArtist[entry.document]])
    const creator = fieldInfo(data.strings[data.setCreator[entry.document]])
    let fields = 0
    for (const token of tokens) {
      fields += TITLE_BOOST * strongMatch(token, title) + ARTIST_BOOST * strongMatch(token, artist) + CREATOR_BOOST * strongMatch(token, creator)
    }
    entry.score += fields / tokens.length
    entry.score += Math.max(similarity(spaced, title.packed), similarity(packedNeedle, title.packed), similarity(packedNeedle, artist.packed + title.packed + creator.packed))
    entry.score += Math.min(entry.stars / 100, POPULARITY_STARS) * POPULARITY_WEIGHT
  }
  ranked.sort((left, right) => right.score - left.score || right.stars - left.stars)
  if (ranked.length > FUZZY_RERANK_LIMIT) ranked.length = FUZZY_RERANK_LIMIT
  for (const entry of ranked) {
    const title = fieldInfo(data.strings[data.setTitle[entry.document]])
    const artist = fieldInfo(data.strings[data.setArtist[entry.document]])
    const creator = fieldInfo(data.strings[data.setCreator[entry.document]])
    entry.score += fuzzyFieldBonus(tokens, title, artist, creator)
  }
  ranked.sort((left, right) => right.score - left.score || right.stars - left.stars)
  return ranked.slice(0, RESULT_LIMIT).map(({ document }) => ({
    id: data.setIds[document],
    artist: data.strings[data.setArtist[document]],
    title: data.strings[data.setTitle[document]],
    creator: data.strings[data.setCreator[document]],
    status: data.setStatus[document],
    year: data.setYear[document] || null,
    diffs: setDiffs(data, document),
  }))
}

export function decodeCorpus(buffer: ArrayBuffer): SearchCorpus {
  const bytes = new Uint8Array(buffer)
  if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'BBS6') {
    throw new Error('Unsupported search artifact')
  }
  const cursor = { offset: 4 }
  const stringCount = decodeVarint(bytes, cursor)
  const termCount = decodeVarint(bytes, cursor)
  const setCount = decodeVarint(bytes, cursor)
  const diffCount = decodeVarint(bytes, cursor)
  const lookupSetCount = decodeVarint(bytes, cursor)
  const lookupDiffCount = decodeVarint(bytes, cursor)
  const lengths = Array.from({ length: 6 }, () => decodeVarint(bytes, cursor))
  let offset = cursor.offset
  const [setSection, diffSection, stringSection, termSection, postingSection, lookupSection] = lengths.map((length) => {
    const section = bytes.subarray(offset, offset + length)
    offset += length
    return section
  })

  const setCursor = { offset: 0 }
  const setIds = decodeColumn(setSection, setCursor, setCount)
  const setArtist = decodeColumn(setSection, setCursor, setCount)
  const setTitle = decodeColumn(setSection, setCursor, setCount)
  const setCreator = decodeColumn(setSection, setCursor, setCount)
  const setMeta = decodeColumn(setSection, setCursor, setCount)
  const setStatus = new Uint8Array(setCount)
  const setYear = new Uint16Array(setCount)
  const setDiffStart = new Uint32Array(setCount)
  const setDiffCount = decodeColumn(setSection, setCursor, setCount)
  const mapIds = new Int32Array(diffCount + lookupDiffCount)
  const mapSetIds = new Uint32Array(mapIds.length)
  let setDiffTotal = 0
  let setId = 0
  for (let index = 0; index < setCount; index++) {
    setId += setIds[index]
    setIds[index] = setId
    setStatus[index] = setMeta[index] & 3
    setDiffStart[index] = setDiffTotal
    setYear[index] = setMeta[index] >>> 2
    mapSetIds.fill(setId, setDiffTotal, setDiffTotal + setDiffCount[index])
    setDiffTotal += setDiffCount[index]
  }

  const diffCursor = { offset: 0 }
  let diffId = 0
  for (let index = 0; index < diffCount; index++) {
    const delta = decodeVarint(diffSection, diffCursor)
    diffId += (delta >>> 1) ^ -(delta & 1)
    mapIds[index] = diffId
  }
  const diffIds = mapIds.subarray(0, diffCount)
  const diffVersions = decodeColumn(diffSection, diffCursor, diffCount)
  const diffStars = decodeColumn(diffSection, diffCursor, diffCount)
  const diffAr = decodeColumn(diffSection, diffCursor, diffCount)
  const diffCs = decodeColumn(diffSection, diffCursor, diffCount)
  const diffOd = decodeColumn(diffSection, diffCursor, diffCount)
  const diffHp = decodeColumn(diffSection, diffCursor, diffCount)
  const diffBpm = decodeColumn(diffSection, diffCursor, diffCount)
  const diffLength = decodeColumn(diffSection, diffCursor, diffCount)

  const lookupCursor = { offset: 0 }
  const lookupSets = decodeColumn(lookupSection, lookupCursor, lookupSetCount)
  const lookupCounts = decodeColumn(lookupSection, lookupCursor, lookupSetCount)
  setId = diffId = 0
  let map = diffCount
  for (let set = 0; set < lookupSetCount; set++) {
    setId += lookupSets[set]
    for (let index = 0; index < lookupCounts[set]; index++, map++) {
      const delta = decodeVarint(lookupSection, lookupCursor)
      diffId += (delta >>> 1) ^ -(delta & 1)
      mapIds[map] = diffId
      mapSetIds[map] = setId
    }
  }

  const postingStarts = new Uint32Array(termCount)
  const postingCounts = new Uint32Array(termCount)
  const postingCursor = { offset: 0 }
  for (let index = 0; index < termCount; index++) {
    postingCounts[index] = decodeVarint(postingSection, postingCursor)
    postingStarts[index] = postingCursor.offset
    for (let document = 0; document < postingCounts[index]; document++) {
      decodeVarint(postingSection, postingCursor)
    }
  }

  return {
    setIds, setArtist, setTitle, setCreator, setStatus, setYear, setDiffStart, setDiffCount,
    diffIds, diffVersions, diffStars, diffAr, diffCs, diffOd, diffHp, diffBpm, diffLength, mapIds, mapSetIds,
    strings: decodeStrings(stringSection, stringCount),
    terms: decodeStrings(termSection, termCount),
    postingStarts, postingCounts, postingBlob: postingSection,
  }
}

function decodeColumn(bytes: Uint8Array, cursor: { offset: number }, count: number): Uint32Array {
  const values = new Uint32Array(count)
  for (let index = 0; index < count; index++) values[index] = decodeVarint(bytes, cursor)
  return values
}

function decodeStrings(bytes: Uint8Array, count: number): string[] {
  const strings: string[] = []
  const cursor = { offset: 0 }
  const decoder = new TextDecoder()
  let value = new Uint8Array(256)
  for (let index = 0; index < count; index++) {
    const prefix = decodeVarint(bytes, cursor)
    const suffix = decodeVarint(bytes, cursor)
    const length = prefix + suffix
    if (length > value.length) {
      const expanded = new Uint8Array(length * 2)
      expanded.set(value)
      value = expanded
    }
    value.set(bytes.subarray(cursor.offset, cursor.offset + suffix), prefix)
    cursor.offset += suffix
    strings.push(decoder.decode(value.subarray(0, length)))
  }
  return strings
}

export function findSetId(corpus: SearchCorpus, beatmapId: number): number | null {
  const index = corpus.mapIds.indexOf(beatmapId)
  return index < 0 ? null : corpus.mapSetIds[index]
}
