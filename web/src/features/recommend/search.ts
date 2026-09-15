const SEARCH_TERM = /[0-9a-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+/g
const COMBINING_MARKS = /[\u0300-\u036f]/g
const REPEATED_LETTER = /([0-9a-z])\1+/g
const EDIT_ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789'

export type SearchDiff = { id: number; version: string; stars: number | null }
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
  mapIds: Int32Array
  mapSetIds: Uint32Array
  strings: string[]
  terms: string[]
  postingStarts: Uint32Array
  postingCounts: Uint32Array
  postingBlob: Uint8Array
}

export function tokenize(text: string): string[] {
  const folded = text.toLowerCase().normalize('NFKD').replace(COMBINING_MARKS, '').replace(REPEATED_LETTER, '$1')
  return folded.match(SEARCH_TERM) ?? []
}

export function editVariants(token: string): Set<string> {
  const variants = new Set<string>()
  for (let index = 0; index < token.length; index++) {
    variants.add(token.slice(0, index) + token.slice(index + 1))
    if (index + 1 < token.length) {
      variants.add(token.slice(0, index) + token[index + 1] + token[index] + token.slice(index + 2))
    }
    for (const letter of EDIT_ALPHABET) {
      variants.add(token.slice(0, index) + letter + token.slice(index + 1))
    }
  }
  for (let index = 0; index <= token.length; index++) {
    for (const letter of EDIT_ALPHABET) {
      variants.add(token.slice(0, index) + letter + token.slice(index))
    }
  }
  return variants
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
    })
  }
  return diffs.sort((left, right) => (right.stars ?? 0) - (left.stars ?? 0))
}

const RESULT_LIMIT = 30
const PREFIX_LIMIT = 64
const TYPO_LIMIT = 80
const EXACT_SCORE = 4
const PREFIX_SCORE = 3
const TYPO_SCORE = 1

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

function termMatches(data: SearchCorpus, token: string): { term: number; score: number }[] {
  const matches: { term: number; score: number }[] = []
  const start = lowerBound(data.terms, token)
  if (data.terms[start] === token) matches.push({ term: start, score: EXACT_SCORE })
  for (let term = start; term < data.terms.length && matches.length < PREFIX_LIMIT; term++) {
    if (!data.terms[term].startsWith(token)) break
    if (term === start && data.terms[start] === token) continue
    matches.push({ term, score: PREFIX_SCORE })
  }
  if (data.terms[start] !== token && token.length >= 4) {
    for (const variant of editVariants(token)) {
      const term = lowerBound(data.terms, variant)
      if (data.terms[term] === variant) matches.push({ term, score: TYPO_SCORE })
      if (matches.length >= TYPO_LIMIT) break
    }
  }
  return matches
}

export function searchCorpus(data: SearchCorpus, query: string): SearchSet[] {
  const tokens = tokenize(query).slice(0, 12)
  const groups = tokens.map((token) => {
    const hits = new Map<number, number>()
    for (const { term, score } of termMatches(data, token)) {
      for (const document of postingDocuments(data, term)) {
        hits.set(document, Math.max(hits.get(document) ?? 0, score))
      }
    }
    return hits
  }).sort((left, right) => left.size - right.size)
  const candidates = [...(groups[0]?.keys() ?? [])].filter((document) => groups.every((group) => group.has(document)))
  const ranked = candidates.map((document) => {
    let score = data.setStatus[document] > 0 ? 1 : 0
    for (const group of groups) score += group.get(document)!
    return { document, score, stars: data.diffStars[data.setDiffStart[document] + data.setDiffCount[document] - 1] }
  })
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
  if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'BBS5') {
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
    diffIds, diffVersions, diffStars, mapIds, mapSetIds,
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
