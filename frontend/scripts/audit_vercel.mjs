import { chromium } from 'playwright'
import path from 'node:path'
import process from 'node:process'
import fs from 'node:fs'

const VERCEL_URL = 'https://hacker-house-goa-task-2.vercel.app/'
const RENDER_BACKEND_URL = 'https://hacker-house-goa-task-2.onrender.com'
const projectRoot = path.resolve(process.cwd(), '..')
const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

function summarize(values) {
  if (!values.length) return { p50: 0, p70: 0, p100: 0, mean: 0, min: 0, max: 0 }
  const s = [...values].sort((a, b) => a - b)
  const n = s.length
  const pct = (p) => {
    const k = (n - 1) * p
    const f = Math.floor(k)
    const c = Math.min(f + 1, n - 1)
    const d = k - f
    return s[f] + d * (s[c] - s[f])
  }
  const sum = s.reduce((a, b) => a + b, 0)
  return {
    p50: pct(0.50),
    p70: pct(0.70),
    p100: s[n - 1],
    mean: sum / n,
    min: s[0],
    max: s[n - 1],
  }
}

const KB_QUERIES = [
  { tag: 'kb', query: 'What is a corporation?' },
  { tag: 'kb', query: 'What is a shareholder?' },
  { tag: 'kb', query: 'How do shareholders vote?' },
  { tag: 'kb', query: 'What causes earthquakes?' },
  { tag: 'kb', query: 'What is photosynthesis?' },
  { tag: 'off', query: "Who won yesterday's cricket match?" },
  { tag: 'kb', query: 'What is the capital of France?' },
  { tag: 'kb', query: 'how long is a marathon' },
  { tag: 'kb', query: 'who invented the telephone' },
  { tag: 'kb', query: 'average body temperature in celsius' },
  { tag: 'kb', query: 'when was the declaration of independence signed' },
  { tag: 'kb', query: 'how many planets are in the solar system' },
  { tag: 'kb', query: 'who wrote romeo and juliet' },
  { tag: 'kb', query: 'what is the boiling point of water' },
  { tag: 'kb', query: 'how does a refrigerator work' },
  { tag: 'kb', query: 'what is the tallest mountain in the world' },
  { tag: 'off', query: 'What is my bank account balance right now?' },
  { tag: 'off', query: 'Give me tonight lottery winning numbers' },
  { tag: 'off', query: 'Book a flight from Goa to Mumbai for tomorrow' },
  { tag: 'off', query: 'What did I eat for breakfast this morning?' },
  { tag: 'kb', query: 'definition of GDP' },
  { tag: 'kb', query: 'symptoms of vitamin D deficiency' },
  { tag: 'off', query: 'Predict next week Bitcoin price exactly' },
]

async function runAudit() {
  console.log('====================================================')
  console.log('STARTING VERCEL PRODUCTION LATENCY AUDIT')
  console.log(`Target Frontend: ${VERCEL_URL}`)
  console.log(`Target Backend:  ${RENDER_BACKEND_URL}`)
  console.log('====================================================\n')

  const browser = await chromium.launch({
    executablePath: fs.existsSync(chromePath) ? chromePath : undefined,
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${path.join(projectRoot, 'data', 'smoke', 'what-is-a-corporation.wav').replaceAll('\\', '/')}`,
    ],
  })

  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  await context.grantPermissions(['microphone'], { origin: VERCEL_URL })
  const page = await context.newPage()

  const networkLog = []
  const consoleErrors = []

  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })

  page.on('request', (req) => {
    networkLog.push({ type: 'request', url: req.url(), method: req.method() })
  })

  page.on('response', (res) => {
    networkLog.push({
      type: 'response',
      url: res.url(),
      status: res.status(),
      cors: res.headers()['access-control-allow-origin'],
    })
  })

  console.log('Step 1: Navigating to Vercel production frontend...')
  const navStart = performance.now()
  await page.goto(VERCEL_URL, { waitUntil: 'networkidle' })
  const navDuration = performance.now() - navStart
  console.log(`✓ Page loaded in ${navDuration.toFixed(1)} ms`)

  // Verify backend connected indicator
  const connectedIndicator = await page.getByText('Backend connected', { exact: true }).isVisible()
  console.log(`✓ Header Backend Connected status: ${connectedIndicator ? 'CONNECTED' : 'NOT CONNECTED'}`)

  // Verify no localhost requests were made
  const localhostCalls = networkLog.filter((n) => n.url.includes('localhost') || n.url.includes('127.0.0.1'))
  console.log(`✓ Localhost network calls: ${localhostCalls.length} (Expected: 0)`)

  // Verify Render backend is called
  const healthCalls = networkLog.filter((n) => n.url.includes('/health'))
  console.log(`✓ Health check target: ${healthCalls.map((h) => h.url).join(', ')}`)

  console.log('\nStep 2: Testing Browser-level RAG queries through Vercel origin context...')

  // Evaluate browser fetch from Vercel origin
  const browserResults = await page.evaluate(async ({ queries, backendUrl }) => {
    async function postQuery(q) {
      const t0 = performance.now()
      const resp = await fetch(`${backendUrl}/api/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      const t1 = performance.now()
      const data = await resp.json()
      return {
        status: resp.status,
        browser_duration_ms: t1 - t0,
        data,
      }
    }

    // Warmup (3 requests)
    const warmups = []
    for (let i = 0; i < 3; i++) {
      warmups.push(await postQuery(queries[i].query))
    }

    // Measured requests
    const measured = []
    for (const item of queries) {
      measured.push({
        tag: item.tag,
        query: item.query,
        result: await postQuery(item.query),
      })
    }

    return { warmups, measured }
  }, { queries: KB_QUERIES, backendUrl: RENDER_BACKEND_URL })

  console.log('✓ 3 Warmups + 23 Measured requests completed from Vercel origin')

  console.log('\n--- WARMUP REQUESTS ---')
  browserResults.warmups.forEach((w, idx) => {
    console.log(`Warmup [${idx + 1}]: browser=${w.browser_duration_ms.toFixed(1)}ms, rag_core=${w.data.latency.rag_core_ms.toFixed(1)}ms, emb=${w.data.latency.embedding_ms.toFixed(1)}ms, bm25=${w.data.latency.bm25_ms.toFixed(2)}ms`)
  })

  console.log('\n--- MEASURED REQUESTS TABLE (FROM VERCEL ORIGIN) ---')
  console.log('#   | Query                     | Browser Wall | RAG Core   | Embedding  | BM25     | Ret. Wall  | Fusion   | Rerank   | Gen      | Ground   | Grounded | Refused')
  console.log('-'.repeat(145))

  const backendRagValues = []
  const backendEmbValues = []
  const backendBm25Values = []
  const backendRetWallValues = []
  const backendTotalValues = []
  const browserWallValues = []

  browserResults.measured.forEach((item, idx) => {
    const r = item.result
    const d = r.data
    const lat = d.latency
    browserWallValues.push(r.browser_duration_ms)
    backendRagValues.push(lat.rag_core_ms)
    backendEmbValues.push(lat.embedding_ms)
    backendBm25Values.push(lat.bm25_ms)
    backendRetWallValues.push(lat.retrieval_wall_ms)
    backendTotalValues.push(lat.total_ms)

    console.log(
      `${String(idx + 1).padStart(2, '0')}  | ` +
      `${item.query.slice(0, 25).padEnd(25, ' ')} | ` +
      `${r.browser_duration_ms.toFixed(1).padStart(9, ' ')}ms | ` +
      `${lat.rag_core_ms.toFixed(1).padStart(7, ' ')}ms | ` +
      `${lat.embedding_ms.toFixed(1).padStart(7, ' ')}ms | ` +
      `${lat.bm25_ms.toFixed(2).padStart(5, ' ')}ms | ` +
      `${lat.retrieval_wall_ms.toFixed(1).padStart(7, ' ')}ms | ` +
      `${lat.fusion_ms.toFixed(2).padStart(5, ' ')}ms | ` +
      `${lat.reranking_ms.toFixed(2).padStart(5, ' ')}ms | ` +
      `${lat.generation_ms.toFixed(2).padStart(5, ' ')}ms | ` +
      `${lat.grounding_ms.toFixed(2).padStart(5, ' ')}ms | ` +
      `${String(d.grounded).padEnd(8, ' ')} | ` +
      `${String(d.refused).padEnd(7, ' ')}`
    )
  })

  const ragStats = summarize(backendRagValues)
  const embStats = summarize(backendEmbValues)
  const bm25Stats = summarize(backendBm25Values)
  const retWallStats = summarize(backendRetWallValues)
  const totalStats = summarize(backendTotalValues)
  const browserStats = summarize(browserWallValues)

  console.log('\n--- PERCENTILES SUMMARY (23 MEASURED REQUESTS) ---')
  console.log('Metric                   | P50 (ms)   | P70 (ms)   | P100 (ms)  | Mean (ms)  | Min (ms)')
  console.log('-'.repeat(80))
  console.log(`Backend RAG Core         | ${ragStats.p50.toFixed(2).padStart(8, ' ')}ms | ${ragStats.p70.toFixed(2).padStart(8, ' ')}ms | ${ragStats.p100.toFixed(2).padStart(8, ' ')}ms | ${ragStats.mean.toFixed(2).padStart(8, ' ')}ms | ${ragStats.min.toFixed(2).padStart(8, ' ')}ms`)
  console.log(`Backend Embedding        | ${embStats.p50.toFixed(2).padStart(8, ' ')}ms | ${embStats.p70.toFixed(2).padStart(8, ' ')}ms | ${embStats.p100.toFixed(2).padStart(8, ' ')}ms | ${embStats.mean.toFixed(2).padStart(8, ' ')}ms | ${embStats.min.toFixed(2).padStart(8, ' ')}ms`)
  console.log(`Backend BM25             | ${bm25Stats.p50.toFixed(2).padStart(8, ' ')}ms | ${bm25Stats.p70.toFixed(2).padStart(8, ' ')}ms | ${bm25Stats.p100.toFixed(2).padStart(8, ' ')}ms | ${bm25Stats.mean.toFixed(2).padStart(8, ' ')}ms | ${bm25Stats.min.toFixed(2).padStart(8, ' ')}ms`)
  console.log(`Backend Retrieval Wall   | ${retWallStats.p50.toFixed(2).padStart(8, ' ')}ms | ${retWallStats.p70.toFixed(2).padStart(8, ' ')}ms | ${retWallStats.p100.toFixed(2).padStart(8, ' ')}ms | ${retWallStats.mean.toFixed(2).padStart(8, ' ')}ms | ${retWallStats.min.toFixed(2).padStart(8, ' ')}ms`)
  console.log(`Backend Total            | ${totalStats.p50.toFixed(2).padStart(8, ' ')}ms | ${totalStats.p70.toFixed(2).padStart(8, ' ')}ms | ${totalStats.p100.toFixed(2).padStart(8, ' ')}ms | ${totalStats.mean.toFixed(2).padStart(8, ' ')}ms | ${totalStats.min.toFixed(2).padStart(8, ' ')}ms`)
  console.log(`Vercel Browser Wall Time | ${browserStats.p50.toFixed(2).padStart(8, ' ')}ms | ${browserStats.p70.toFixed(2).padStart(8, ' ')}ms | ${browserStats.p100.toFixed(2).padStart(8, ' ')}ms | ${browserStats.mean.toFixed(2).padStart(8, ' ')}ms | ${browserStats.min.toFixed(2).padStart(8, ' ')}ms`)

  console.log('\nStep 3: Testing Real Voice Recording UI Flow on Vercel frontend...')

  // Click microphone button to start recording
  console.log('Testing Grounded Voice Flow: "What is a corporation?"...')
  const micButton = page.getByRole('button', { name: 'Start voice recording' })
  await micButton.click()

  const stopButton = page.getByRole('button', { name: 'Stop voice recording' })
  await stopButton.waitFor({ state: 'visible' })
  await page.waitForTimeout(3000) // Record audio for 3s

  const voiceUploadStart = performance.now()
  await stopButton.click()

  // Wait for result UI to render
  await page.getByText('“what is a corporation”', { exact: false }).waitFor({ timeout: 45000 })
  const voiceUploadDuration = performance.now() - voiceUploadStart
  console.log(`✓ Voice query resolved in ${voiceUploadDuration.toFixed(1)} ms`)

  // Inspect rendered UI values
  const displayedRagLatency = await page.locator('.result-card').filter({ hasText: 'RAG latency' }).locator('p.font-mono').first().innerText()
  const displayedTargetStatus = await page.getByText('Under 200 ms target', { exact: true }).isVisible()
  const displayedAnswer = await page.getByText('McDonald\'s Corporation', { exact: false }).isVisible() || await page.getByText('A corporation is a', { exact: false }).isVisible()
  const displayedEvidenceSummary = await page.getByText(/Grounded using \d+ retrieved passage/).innerText()

  console.log(`✓ Rendered RAG Latency Metric: "${displayedRagLatency}"`)
  console.log(`✓ Target Indicator Visible: ${displayedTargetStatus ? '✓ Under 200 ms target' : 'Above target'}`)
  console.log(`✓ Answer Rendered: ${displayedAnswer}`)
  console.log(`✓ Evidence Summary: "${displayedEvidenceSummary}"`)

  // Test evidence modal
  const viewEvidenceBtn = page.getByRole('button', { name: 'View evidence' })
  await viewEvidenceBtn.click()
  const dialogVisible = await page.getByRole('dialog', { name: 'Retrieved passages' }).isVisible()
  console.log(`✓ Evidence Modal Opened: ${dialogVisible}`)
  await page.getByRole('button', { name: 'Close' }).click()

  console.log('\nAuditing Voice API Response payload...')
  const voiceResponses = networkLog.filter((n) => n.url.includes('/api/voice/query') && n.type === 'response')
  console.log(`Voice API responses captured: ${voiceResponses.length}`)
  if (voiceResponses.length > 0) {
    console.log(`Voice response status: ${voiceResponses[0].status}`)
    console.log(`Voice response CORS: ${voiceResponses[0].cors}`)
  }

  await browser.close()

  console.log('\n====================================================')
  console.log('AUDIT COMPLETED SUCCESSFULLY')
  console.log('====================================================')

  return {
    ragStats,
    embStats,
    bm25Stats,
    retWallStats,
    totalStats,
    browserStats,
    voiceUploadDuration,
    displayedRagLatency,
    displayedTargetStatus,
    consoleErrors,
    localhostCalls: localhostCalls.length,
  }
}

runAudit()
  .then(() => {
    process.exit(0)
  })
  .catch((err) => {
    console.error('Audit failed with error:', err)
    process.exit(1)
  })
