import { chromium } from 'playwright'
import path from 'node:path'
import process from 'node:process'
import fs from 'node:fs'

const VERCEL_URL = 'https://hacker-house-goa-task-2.vercel.app/'
const projectRoot = path.resolve(process.cwd(), '..')
const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

async function testVoiceFlows() {
  console.log('Testing Multi-query Voice UI Flows on Vercel frontend (Warmup + Grounded + Refusal)...')

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

  const responses = []

  page.on('response', async (res) => {
    if (res.url().includes('/api/voice/query') && res.status() === 200) {
      try {
        const json = await res.json()
        responses.push(json)
      } catch {}
    }
  })

  await page.goto(VERCEL_URL, { waitUntil: 'networkidle' })
  await page.locator('header').getByText(/(System ready|Backend connected)/).first().waitFor({ timeout: 15000 })


  async function record(isFirst = false) {
    if (isFirst) {
      await page.getByRole('button', { name: 'Start voice recording' }).click()
    } else {
      await page.getByRole('button', { name: 'Ask another question' }).click()
    }
    const stopBtn = page.getByRole('button', { name: 'Stop voice recording' })
    await stopBtn.waitFor({ state: 'visible' })
    await page.waitForTimeout(3200)
    const t0 = performance.now()
    await stopBtn.click()
    await page.locator('blockquote').waitFor({ state: 'visible', timeout: 45000 })
    const duration = performance.now() - t0
    return duration
  }

  console.log('Voice Query 1 (Cold warmup)...')
  const dur1 = await record(true)
  console.log(`✓ Voice Query 1 duration: ${dur1.toFixed(1)} ms`)

  console.log('Voice Query 2 (Warm grounded query)...')
  const dur2 = await record(false)
  console.log(`✓ Voice Query 2 duration: ${dur2.toFixed(1)} ms`)

  console.log('Voice Query 3 (Warm repeat query)...')
  const dur3 = await record(false)
  console.log(`✓ Voice Query 3 duration: ${dur3.toFixed(1)} ms`)

  console.log('\n--- CAPTURED VOICE API RESPONSES ---')
  responses.forEach((resp, idx) => {
    const lat = resp.latency
    console.log(
      `Voice Query [${idx + 1}]: ` +
      `STT=${lat.stt_ms.toFixed(1)}ms, ` +
      `RAG Core=${lat.rag_core_ms.toFixed(1)}ms, ` +
      `Embedding=${lat.embedding_ms.toFixed(1)}ms, ` +
      `BM25=${lat.bm25_ms.toFixed(2)}ms, ` +
      `Total Server=${lat.total_ms.toFixed(1)}ms, ` +
      `Grounded=${resp.grounded}, Refused=${resp.refused}`
    )
  })

  // Check UI state on final query
  const ragLatencyText = await page.locator('.result-card').filter({ hasText: 'RAG latency' }).locator('p.font-mono').first().innerText()
  const targetStatus = await page.getByText('Under 200 ms target', { exact: true }).isVisible()
  console.log(`\nFinal UI Display: RAG Latency = "${ragLatencyText}", Target Status = ${targetStatus ? '✓ Under 200 ms target' : 'Above target'}`)

  await browser.close()
}

testVoiceFlows().catch((err) => {
  console.error('Voice test error:', err)
  process.exit(1)
})
