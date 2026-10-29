import test from 'node:test'
import assert from 'node:assert/strict'
import { ref } from 'vue'
import { useProductChat } from '../src/composables/useProductChat.js'
import { sendChatMessage } from '../src/api/chat.js'

const ids = ['vivo_x500_pro', 'vivo_x500', 'vivo_x500_pro_max']
function setup(send) {
  const selectedId = ref(ids[0])
  const products = ref(ids.map(id => ({ id })))
  return { selectedId, ...useProductChat({ selectedId, products, send }) }
}
const response = (product_id, answer = '回答', session_id = 'same-session') => ({
  product_id, answer, session_id, citations: []
})

test('四轮追问沿用会话，服务端响应切换商品，保留回答及引用', async () => {
  const calls = []
  const expected = [ids[0], ids[1], ids[2], ids[0]]
  const chat = setup(async payload => {
    calls.push(payload)
    const result = response(expected[calls.length - 1])
    if (calls.length < 4) result.citations = [{ number: 1, snippet: '原文' }]
    return result
  })
  for (const question of ['无线充电？', '那普通版呢？', 'Pro Max 呢？', '回到 X500 Pro，备注是什么？']) {
    await chat.sendMessage(question)
    assert.equal(chat.selectedId.value, expected[calls.length - 1])
  }
  assert.deepEqual(calls.map(c => c.sessionId), [null, 'same-session', 'same-session', 'same-session'])
  assert.deepEqual(calls.map(c => c.productId), [ids[0], ids[0], ids[1], ids[2]])
  assert.ok(calls.every(c => c.followConversation === true))
  assert.equal(chat.messages.value.length, 8)
  assert.deepEqual(chat.messages.value[1].citations, [{ number: 1, snippet: '原文' }])
  assert.equal(chat.messages.value.at(-1).productId, ids[0])
})

test('手动选择商品保留会话，只有新对话清除会话及消息', async () => {
  const calls = []
  const chat = setup(async payload => {
    calls.push(payload)
    return response(payload.productId)
  })
  await chat.sendMessage('记住我的备注')
  chat.selectedId.value = ids[1]
  await chat.sendMessage('这款有哪些颜色')
  assert.equal(calls[1].sessionId, 'same-session')
  assert.equal(chat.messages.value.length, 4)
  assert.equal(chat.newConversation(), true)
  assert.equal(chat.sessionId.value, null)
  assert.equal(chat.messages.value.length, 0)
  assert.equal(chat.selectedId.value, ids[1])
  await chat.sendMessage('新的问题')
  assert.equal(calls[2].sessionId, null)
})

test('请求期间阻止重复发送及重置', async () => {
  let finish
  const chat = setup(() => new Promise(resolve => { finish = resolve }))
  const first = chat.sendMessage('一个问题')
  assert.equal(chat.isLoading.value, true)
  assert.equal(await chat.sendMessage('重复问题'), null)
  assert.equal(chat.newConversation(), false)
  finish(response(ids[0]))
  await first
  assert.equal(chat.isLoading.value, false)
  assert.equal(chat.messages.value.length, 2)
})

test('澄清不擅自切商品，错误商品响应不覆盖已有会话', async () => {
  let result = response(null, '请明确商品')
  const chat = setup(async () => result)
  await chat.sendMessage('X999 呢？')
  assert.equal(chat.selectedId.value, ids[0])
  assert.equal(chat.messages.value.at(-1).productId, null)
  result = response('unknown-product', '错误的商品', 'wrong-session')
  assert.equal(await chat.sendMessage('再问一次'), null)
  assert.equal(chat.sessionId.value, 'same-session')
  assert.equal(chat.selectedId.value, ids[0])
  assert.equal(chat.messages.value.at(-1).role, 'error')
})

test('网络失败释放发送状态，不伪造回答或会话', async () => {
  const chat = setup(async () => { throw new Error('服务暂时无法连接') })
  await chat.sendMessage('测试问题')
  assert.equal(chat.isLoading.value, false)
  assert.equal(chat.sessionId.value, null)
  assert.equal(chat.messages.value.at(-1).content, '服务暂时无法连接')
  assert.equal(chat.messages.value.at(-1).role, 'error')
})

test('空问题与无商品不发起请求', async () => {
  const chat = setup(() => { throw new Error('不应调用') })
  assert.equal(await chat.sendMessage('  '), null)
  chat.selectedId.value = ''
  assert.equal(await chat.sendMessage('问题'), null)
  assert.equal(chat.messages.value.length, 0)
})

test('正式聊天请求启用会话承接，并正确序列化 JSON', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options })
    return { ok: true, json: async () => response(ids[1]) }
  })
  await sendChatMessage({ productId: ids[0], message: '那普通版呢？', sessionId: 'same-session' })
  assert.equal(calls[0].url, '/api/chat')
  assert.equal(calls[0].options.method, 'POST')
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    product_id: ids[0], message: '那普通版呢？', session_id: 'same-session', follow_conversation: true
  })
})

test('接口错误传递校验信息，非 JSON 错误保留 HTTP 状态', async t => {
  t.mock.method(globalThis, 'fetch', async () => ({ ok: false, status: 404, json: async () => ({ detail: '当前商品不存在。' }) }))
  await assert.rejects(sendChatMessage({ message: '问题' }), /当前商品不存在/)
  globalThis.fetch = async () => ({ ok: false, status: 502, json: async () => { throw new Error('HTML') } })
  await assert.rejects(sendChatMessage({ message: '问题' }), /HTTP 502/)
})
