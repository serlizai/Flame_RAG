import { ref } from 'vue'
import { sendChatMessage } from '../api/chat.js'

// 商品切换沿用会话编号，商品历史的隔离由后端负责。
export function useProductChat({ selectedId, products, send = sendChatMessage }) {
  const messages = ref([])
  const sessionId = ref(null)
  const isLoading = ref(false)

  async function sendMessage(text) {
    const question = String(text).trim()
    if (!question || !selectedId.value || isLoading.value) return null

    const productId = selectedId.value
    messages.value.push({ role: 'user', content: question, productId })
    isLoading.value = true

    try {
      const result = await send({
        productId,
        message: question,
        sessionId: sessionId.value,
        followConversation: true
      })
      if (
        typeof result.answer !== 'string' || !result.answer.trim() ||
        typeof result.session_id !== 'string' || !result.session_id.trim() ||
        (result.product_id !== null && !products.value.some(p => p.id === result.product_id))
      ) {
        throw new Error('问答服务返回格式不正确，请稍后重试。')
      }

      sessionId.value = result.session_id
      // 后端确认的商品身份用于同步卡片；澄清或多商品问题不擅自选择商品。
      if (result.product_id !== null) selectedId.value = result.product_id
      messages.value.push({
        role: 'assistant',
        content: result.answer,
        productId: result.product_id,
        citations: Array.isArray(result.citations) ? result.citations : []
      })
      return result
    } catch (error) {
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
