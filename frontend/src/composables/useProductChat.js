import { ref } from 'vue'
import { streamChatMessage } from '../api/chat.js'

// 商品切换沿用会话编号，商品历史的隔离由后端负责。
export function useProductChat({ selectedId, products, send = streamChatMessage }) {
  const messages = ref([])
  const sessionId = ref(null)
  const isLoading = ref(false)

  async function sendMessage(text) {
    const question = String(text).trim()
    if (!question || !selectedId.value || isLoading.value) return null

    const productId = selectedId.value
    messages.value.push({ role: 'user', content: question, productId })
    messages.value.push({
      role: 'assistant', content: '', productId, citations: [],
      streaming: true, interrupted: false, status: '正在准备回答…'
    })
    const replyIndex = messages.value.length - 1
    const reply = messages.value[replyIndex]
    isLoading.value = true

    function checkIdentity(data) {
      if (
        typeof data.session_id !== 'string' || !data.session_id.trim() ||
        (data.product_id !== null && !products.value.some(p => p.id === data.product_id))
      ) throw new Error('问答服务返回格式不正确，请稍后重试。')
    }

    try {
      const result = await send({
        productId,
        message: question,
        sessionId: sessionId.value,
        followConversation: true,
        onEvent(event, data) {
          if (event === 'meta') {
            checkIdentity(data)
            sessionId.value = data.session_id
            reply.productId = data.product_id
            if (data.product_id !== null) selectedId.value = data.product_id
          } else if (event === 'status' && typeof data.message === 'string') {
            reply.status = data.message
          } else if (event === 'delta') {
            if (typeof data.text !== 'string') throw new Error('流式回答格式不正确。')
            reply.content += data.text
            reply.status = '正在生成…'
          }
        }
      })
      if (
        typeof result.answer !== 'string' || !result.answer.trim() ||
        typeof result.session_id !== 'string' || !result.session_id.trim() ||
        (result.product_id !== null && !products.value.some(p => p.id === result.product_id))
      ) {
        throw new Error('问答服务返回格式不正确，请稍后重试。')
      }
      checkIdentity(result)

      sessionId.value = result.session_id
      // 后端确认的商品身份用于同步卡片；澄清或多商品问题不擅自选择商品。
      if (result.product_id !== null) selectedId.value = result.product_id
      Object.assign(reply, {
        content: result.answer,
        productId: result.product_id,
        citations: Array.isArray(result.citations) ? result.citations : [],
        streaming: false, status: ''
      })
      return result
    } catch (error) {
      reply.streaming = false
      reply.interrupted = true
      reply.status = ''
      if (!reply.content) messages.value.splice(replyIndex, 1)
      messages.value.push({
        role: 'error',
        content: error instanceof Error
          ? error.message
          : '问答服务暂时无法连接，请稍后重试。',
        productId
      })
      return null
    } finally {
      isLoading.value = false
    }
  }

  function newConversation() {
    if (isLoading.value) return false
    messages.value = []
    sessionId.value = null
    return true
  }

  return { messages, sessionId, isLoading, sendMessage, newConversation }
}
