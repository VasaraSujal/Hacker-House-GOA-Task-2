import { chromium } from 'playwright'
import path from 'node:path'
import process from 'node:process'

const projectRoot = path.resolve(process.cwd(), '..')
const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const frontendUrl = 'http://127.0.0.1:5173'

async function runVoiceCase({ audioFile, expectedTranscript, expectedState, screenshot, repeat = false, viewport }) {
  const browser = await chromium.launch({
    executablePath: chromePath,
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${path.join(projectRoot, 'data', 'smoke', audioFile).replaceAll('\\', '/')}`,
    ],
  })
  const context = await browser.newContext({ viewport })
  await context.grantPermissions(['microphone'], { origin: frontendUrl })
  const page = await context.newPage()
  const consoleErrors = []
  const voiceResponses = []

  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  page.on('response', async (response) => {
    if (response.url().endsWith('/api/voice/query')) {
      voiceResponses.push({
        status: response.status(),
        cors: response.headers()['access-control-allow-origin'],
        body: await response.text(),
      })
    }
  })
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    window.__voiceStreams = []
    navigator.mediaDevices.getUserMedia = async (...args) => {
      const stream = await original(...args)
      window.__voiceStreams.push(stream)
      return stream
    }
  })

  const ask = async (firstRun) => {
    if (firstRun) {
      await page.getByRole('button', { name: 'Start voice recording' }).click()
    } else {
      await page.getByRole('button', { name: 'Ask another question' }).click()
    }
    await page.getByRole('button', { name: 'Stop voice recording' }).waitFor()
    await page.waitForTimeout(3200)
    await page.getByRole('button', { name: 'Stop voice recording' }).click()
    const outcome = await Promise.race([
      page.getByText(expectedState, { exact: true }).waitFor({ timeout: 45_000 }).then(() => 'result'),
      page.getByRole('alert').waitFor({ timeout: 45_000 }).then(() => 'error'),
    ])
    if (outcome === 'error') {
      throw new Error(`UI error: ${await page.getByRole('alert').innerText()}; responses=${JSON.stringify(voiceResponses)}`)
    }
  }

  await page.goto(frontendUrl)
  await page.locator('header').getByText(/(System ready|Backend connected)/).first().waitFor({ timeout: 15_000 })
  await ask(true)
  if (repeat) await ask(false)

  const transcript = await page.locator('blockquote').innerText()
  const tracksReleased = await page.evaluate(() =>
    window.__voiceStreams.every((stream) => stream.getTracks().every((track) => track.readyState === 'ended')),
  )
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    offenders: [...document.querySelectorAll('*')]
      .filter((element) => element.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
      .slice(0, 5)
      .map((element) => ({
        tag: element.tagName,
        className: element.className,
        right: Math.round(element.getBoundingClientRect().right),
      })),
  }))
  const noOverflow = overflow.scrollWidth <= overflow.clientWidth
  const sourceCountText = await page.getByText(/\d+ retrieved passage/).innerText()
  const performanceVisible = await page.getByText('RAG latency', { exact: true }).first().isVisible()
  await page.screenshot({ path: path.join(projectRoot, 'benchmarks', screenshot), fullPage: true })
  await browser.close()

  if (!transcript.toLowerCase().includes(expectedTranscript.toLowerCase())) {
    throw new Error(`Unexpected transcript: ${transcript}`)
  }
  if (!tracksReleased) throw new Error('Microphone tracks were not released')
  if (!noOverflow) throw new Error(`Page has horizontal overflow: ${JSON.stringify(overflow)}`)
  if (!performanceVisible) throw new Error('Latency panel was not visible')
  if (voiceResponses.some(({ status, cors }) => status !== 200 || cors !== frontendUrl)) {
    throw new Error(`Voice request or CORS failure: ${JSON.stringify(voiceResponses)}`)
  }
  if (consoleErrors.length) throw new Error(`Browser console errors: ${consoleErrors.join('; ')}`)

  return { transcript, sourceCountText, tracksReleased, noOverflow, voiceResponses }
}

const relevant = process.env.VOICE_CASE === 'refusal' ? null : await runVoiceCase({
    audioFile: 'what-is-a-corporation.wav',
    expectedTranscript: 'what is a corporation',
    expectedState: 'Grounded answer',
    screenshot: 'stage4-browser-relevant.png',
    repeat: true,
    viewport: { width: 1440, height: 1000 },
  })

const refusal = process.env.VOICE_CASE === 'relevant' ? null : await runVoiceCase({
    audioFile: 'unsupported-question.wav',
    expectedTranscript: "who won yesterday's cricket match",
    expectedState: 'Knowledge-base refusal',
    screenshot: 'stage4-browser-refusal-mobile.png',
    viewport: { width: 390, height: 844 },
  })

console.log(JSON.stringify({ relevant, refusal }, null, 2))
