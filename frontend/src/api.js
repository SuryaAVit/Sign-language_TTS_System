const BASE = '/api'

export async function startSession() {
  const res = await fetch(`${BASE}/session/start`, { method:'POST' })
  return res.json()
}

export async function endSession(session_id) {
  await fetch(`${BASE}/session/end`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ session_id })
  })
}

// ── NEW: send pre-computed landmarks from browser MediaPipe ────────────────
// Flask only needs to run the ML model — no frame decoding, no OpenCV
export async function recognizeFromLandmarks(landmarks, session_id) {
  const res = await fetch(`${BASE}/gesture/classify`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ landmarks, session_id })
  })
  if (!res.ok) return { detected:false }
  return res.json()
}

export async function updateTranscript(session_id, text) {
  await fetch(`${BASE}/transcript/update`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ session_id, text })
  })
}

export async function generateTTS(text) {
  const res = await fetch(`${BASE}/tts`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text, online:true })
  })
  if (!res.ok) throw new Error('TTS failed')
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

export async function exportPDF(text, session_id) {
  const res = await fetch(`${BASE}/export/pdf`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text, session_id })
  })
  if (!res.ok) throw new Error('PDF failed')
  const blob = await res.blob()
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href=url; a.download='transcript.pdf'; a.click()
  URL.revokeObjectURL(url)
}

export async function exportWord(text, session_id) {
  const res = await fetch(`${BASE}/export/word`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text, session_id })
  })
  if (!res.ok) throw new Error('Word failed')
  const blob = await res.blob()
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href=url; a.download='transcript.docx'; a.click()
  URL.revokeObjectURL(url)
}

export async function getHistory(session_id) {
  const res = await fetch(`${BASE}/session/${session_id}/history`)
  return res.json()
}