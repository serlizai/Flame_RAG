// 调用 FastAPI 的正式 AI 问答接口
export async function sendChatMessage({
  productId,
  message,
  sessionId = null
}) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      product_id: productId,
      message,
      session_id: sessionId
    })
  })

  if (!response.ok) {
    let errorMessage = `请求失败：HTTP ${response.status}`

    try {
      const errorData = await response.json()

      if (typeof errorData.detail === 'string') {
        errorMessage = errorData.detail
      }
    } catch {
      // 如果后端返回的不是 JSON，保留 HTTP 错误信息
    }

    throw new Error(errorMessage)
  }

  return response.json()
}