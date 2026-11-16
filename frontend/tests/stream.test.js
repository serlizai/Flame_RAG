import test from 'node:test'
import assert from 'node:assert/strict'
import { ref } from 'vue'
import { streamChatMessage } from '../src/api/chat.js'
import { useProductChat } from '../src/composables/useProductChat.js'

const productId = 'vivo_x500_pro'
const final = { answer: '校验后。[1]', citations: [{ number: 1 }], product_id: productId, session_id: 'stream-session' }
const frame = (name, data) => `event: ${name}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`

test('UTF-8 中文跨字节分块及 CRLF 事件跨块仍能解析，done 决定最终答案', async t => {
  const encoded = new TextEncoder().encode(
    ': ping\r\n\r\n' + frame('meta', final) + frame('delta', { text: '中文\n内容' }) + frame('done', final)
  )
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options })
    let offset = 0
    return new Response(new ReadableStream({
      pull(controller) {
        if (offset === encoded.length) controller.close()
        else controller.enqueue(encoded.slice(offset, ++offset))
      }
    }), { headers: { 'content-type': 'text/event-stream; charset=utf-8' } })
  })
  const received = []
  const result = await streamChatMessage({ productId, message: '问题', onEvent: (...args) => received.push(args) })
  assert.deepEqual(result, final)
  assert.equal(calls[0].url, '/api/chat/stream')
  assert.equal(calls[0].options.method, 'POST')
  assert.deepEqual(received.map(([name]) => name), ['meta', 'delta', 'done'])
  assert.equal(received[1][1].text, '中文\n内容')
})

test('多个事件一次到达，错误事件和缺少 done 均不能作为成功结果', async t => {
  const respond = text => new Response(text, { headers: { 'content-type': 'text/event-stream' } })
  t.mock.method(globalThis, 'fetch', async () => respond(frame('delta', { text: '半截回答' }) + frame('error', { message: '服务异常' })))
  await assert.rejects(streamChatMessage({ message: '问题' }), /服务异常/)
  globalThis.fetch = async () => respond(frame('delta', { text: '半截回答' }))
  await assert.rejects(streamChatMessage({ message: '问题' }), /尚未完成/)
  globalThis.fetch = async () => new Response('{}', { headers: { 'content-type': 'application/json' } })
  await assert.rejects(streamChatMessage({ message: '问题' }), /未返回流式响应/)
})

test('HTTP 校验错误及请求超时释放连接', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ detail: '当前商品不存在。' }), { status: 404 }))
  await assert.rejects(streamChatMessage({ message: '问题' }), /当前商品不存在/)
  globalThis.fetch = (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
  })
  await assert.rejects(streamChatMessage({ message: '问题', timeoutMs: 10 }), /超时/)
})

test('模型未结束时同一个气泡持续增长，最终校验替换初稿并添加引用', async () => {
  let finish
  const selectedId = ref(productId)
  const chat = useProductChat({
    selectedId, products: ref([{ id: productId }]),
    send: async ({ onEvent }) => {
      onEvent('meta', final)
      onEvent('delta', { text: '第一段' })
      await new Promise(resolve => { finish = resolve })
      onEvent('delta', { text: '第二段[999]' })
      return final
    }
  })
  const pending = chat.sendMessage('问题')
  assert.equal(chat.isLoading.value, true)
  assert.equal(chat.messages.value.length, 2)
  assert.equal(chat.messages.value[1].content, '第一段')
  assert.equal(chat.messages.value[1].streaming, true)
  assert.deepEqual(chat.messages.value[1].citations, [])
  finish()
  await pending
  assert.equal(chat.messages.value.length, 2)
  assert.equal(chat.messages.value[1].content, final.answer)
  assert.deepEqual(chat.messages.value[1].citations, final.citations)
  assert.equal(chat.messages.value[1].streaming, false)
})

test('断线保留已收初稿并标记中断，沿用已确认的会话且不自动重发', async () => {
  let calls = 0
  const chat = useProductChat({
    selectedId: ref(productId), products: ref([{ id: productId }]),
    send: async ({ onEvent }) => {
      calls++
      onEvent('meta', final)
      onEvent('delta', { text: '未完成内容' })
      throw new Error('连接已中断')
    }
  })
  assert.equal(await chat.sendMessage('问题'), null)
  assert.equal(chat.messages.value[1].interrupted, true)
  assert.equal(chat.messages.value.at(-1).role, 'error')
  assert.equal(chat.sessionId.value, final.session_id)
  assert.equal(chat.isLoading.value, false)
  assert.equal(calls, 1)
})
