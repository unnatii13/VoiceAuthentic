import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE = '/api'

const initialDashboard = {
  total_scans: 0,
  deepfakes_detected: 0,
  real_voices: 0,
}

function formatPercent(value) {
  if (typeof value !== 'number') return '0%'
  return `${Math.round(value * 100)}%`
}

function severityClass(severity = '') {
  return severity.toLowerCase()
}

function App() {
  const [dashboard, setDashboard] = useState(initialDashboard)
  const [incidents, setIncidents] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [result, setResult] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [apiError, setApiError] = useState('')
  const inputRef = useRef(null)

  const fakeRate = useMemo(() => {
    if (!dashboard.total_scans) return 0
    return dashboard.deepfakes_detected / dashboard.total_scans
  }, [dashboard])

  const latestIncidents = useMemo(() => incidents.slice(-5).reverse(), [incidents])

  async function loadDashboard() {
    try {
      const [dashboardResponse, incidentsResponse] = await Promise.all([
        fetch(`${API_BASE}/dashboard`),
        fetch(`${API_BASE}/incidents`),
      ])

      if (!dashboardResponse.ok || !incidentsResponse.ok) {
        throw new Error('Backend is not responding correctly.')
      }

      setDashboard(await dashboardResponse.json())
      setIncidents(await incidentsResponse.json())
      setApiError('')
    } catch (error) {
      setApiError(error.message)
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadDashboard()
  }, [])

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  function pickFile(file) {
    if (!file) return
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setSelectedFile(file)
    setAudioUrl(URL.createObjectURL(file))
    setResult(null)
    setStatusMessage('Audio queued for forensic scan.')
    setApiError('')
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    pickFile(event.dataTransfer.files?.[0])
  }

  async function analyzeFile() {
    if (!selectedFile) {
      setStatusMessage('Choose a WAV audio file before starting analysis.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    setIsAnalyzing(true)
    setStatusMessage('Extracting MFCC, pitch, spectral contrast, and energy signals.')
    setApiError('')

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed. Check that the FastAPI backend is running.')
      }

      const data = await response.json()
      setResult(data)
      setStatusMessage('Analysis complete. Review the risk report below.')
      await loadDashboard()
    } catch (error) {
      setApiError(error.message)
      setStatusMessage('Analysis could not be completed.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <main className="app-shell">
      <nav className="topbar" aria-label="Main navigation">
        <a className="brand" href="#scanner" aria-label="VoiceAuthentic home">
          <span className="brand-mark">VA</span>
          <span>VoiceAuthentic</span>
        </a>
        <div className="nav-links">
          <a href="#scanner">Scanner</a>
          <a href="#dashboard">Dashboard</a>
          <a href="#incidents">Incidents</a>
        </div>
      </nav>

      <section className="hero-section" id="scanner">
        <div className="hero-copy">
          <span className="eyebrow">Deepfake Voice Scammer Firewall</span>
          <h1>Verify voices before trust becomes a vulnerability.</h1>
          <p>
            Upload an audio sample and VoiceAuthentic scores the probability of
            synthetic speech using the backend model, confidence, and threat
            severity pipeline.
          </p>
          <div className="hero-actions">
            <button className="primary-action" type="button" onClick={() => inputRef.current?.click()}>
              Select audio
            </button>
            <a className="secondary-action" href="#dashboard">
              View signals
            </a>
          </div>
        </div>

        <section className="scanner-panel" aria-label="Audio scanner">
          <div className="panel-header">
            <div>
              <span className="panel-kicker">Forensic input</span>
              <h2>Voice scan</h2>
            </div>
            <span className="live-pill">Model ready</span>
          </div>

          <div
            className={`drop-zone ${isDragging ? 'is-dragging' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="audio/*,.wav"
              onChange={(event) => pickFile(event.target.files?.[0])}
            />
            <div className="scan-orbit" aria-hidden="true">
              <span></span>
            </div>
            <strong>{selectedFile ? selectedFile.name : 'Drop a voice sample here'}</strong>
            <p>{selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB ready to scan` : 'WAV and common audio files are supported.'}</p>
          </div>

          {audioUrl && (
            <audio className="audio-preview" controls src={audioUrl}>
              Your browser does not support the audio element.
            </audio>
          )}

          <button
            className="analyze-button"
            type="button"
            disabled={isAnalyzing}
            onClick={analyzeFile}
          >
            {isAnalyzing ? 'Analyzing audio...' : 'Run deepfake analysis'}
          </button>

          {(statusMessage || apiError) && (
            <p className={`status-line ${apiError ? 'is-error' : ''}`}>
              {apiError || statusMessage}
            </p>
          )}
        </section>
      </section>

      <section className="dashboard-grid" id="dashboard" aria-label="Detection dashboard">
        <article className="signal-card accent">
          <span>Total scans</span>
          <strong>{dashboard.total_scans}</strong>
          <p>Samples processed by the model pipeline.</p>
        </article>
        <article className="signal-card danger">
          <span>Deepfakes detected</span>
          <strong>{dashboard.deepfakes_detected}</strong>
          <p>{formatPercent(fakeRate)} of all logged scans flagged synthetic.</p>
        </article>
        <article className="signal-card safe">
          <span>Authentic voices</span>
          <strong>{dashboard.real_voices}</strong>
          <p>Human speech decisions from recent analyses.</p>
        </article>
      </section>

      {result && (
        <section className="result-panel" aria-label="Latest analysis result">
          <div className="result-score">
            <span className={`verdict ${result.prediction?.toLowerCase()}`}>
              {result.prediction}
            </span>
            <strong>{result.threat_score}</strong>
            <p>Threat score</p>
          </div>
          <div className="result-details">
            <span className={`severity ${severityClass(result.severity)}`}>
              {result.severity}
            </span>
            <h2>{result.filename}</h2>
            <p>Confidence: {formatPercent(result.confidence)}</p>
            <div className="reason-list">
              {(result.reasons || []).map((reason) => (
                <span key={reason}>{reason}</span>
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="incidents-section" id="incidents">
        <div className="section-heading">
          <span className="eyebrow">Logged evidence</span>
          <h2>Recent incident trail</h2>
        </div>

        <div className="incident-table" role="table" aria-label="Recent incident history">
          <div className="incident-row incident-head" role="row">
            <span role="columnheader">File</span>
            <span role="columnheader">Verdict</span>
            <span role="columnheader">Confidence</span>
            <span role="columnheader">Severity</span>
            <span role="columnheader">Time</span>
          </div>
          {latestIncidents.length > 0 ? (
            latestIncidents.map((incident) => (
              <div className="incident-row" role="row" key={`${incident.timestamp}-${incident.filename}`}>
                <span role="cell">{incident.filename}</span>
                <span role="cell" className={`verdict-text ${incident.prediction?.toLowerCase()}`}>
                  {incident.prediction}
                </span>
                <span role="cell">{formatPercent(incident.confidence)}</span>
                <span role="cell" className={`severity-text ${severityClass(incident.severity)}`}>
                  {incident.severity}
                </span>
                <span role="cell">{incident.timestamp}</span>
              </div>
            ))
          ) : (
            <div className="empty-state">No scan incidents have been logged yet.</div>
          )}
        </div>
      </section>
    </main>
  )
}

export default App
