// 调用 FastAPI 的正式 AI 问答接口
export async function sendChatMessage({
  productId,
  message,
  sessionId = null,
  followConversation = true
}) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      product_id: productId,
      message,
      session_id: sessionId,
      follow_conversation: followConversation
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

// POST 携带会话信息，持续读取 SSE。仅 done 表示本轮回答已经完成。
export async function streamChatMessage({
  productId, message, sessionId = null, followConversation = true,
  onEvent = () => {}, timeoutMs = 300000
}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  let reader
  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      signal: controller.signal,
      body: JSON.stringify({
        product_id: productId, message, session_id: sessionId,
        follow_conversation: followConversation
      })
    })
    if (!response.ok) {
      let detail
      try { detail = (await response.json()).detail } catch { /* 保留 HTTP 状态 */ }
      throw new Error(typeof detail === 'string' ? detail : `请求失败：HTTP ${response.status}`)
    }
    if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) {
      throw new Error('问答服务未返回流式响应，请检查后端是否已更新。')
    }
    reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8', { fatal: true })
    let buffer = ''
    let eventName = 'message'
    let dataLines = []
    while (true) {
      const { value, done } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      let newline
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).replace(/\r$/, '')
        buffer = buffer.slice(newline + 1)
        if (line === '') {
          if (dataLines.length) {
            const data = JSON.parse(dataLines.join('\n'))
            if (eventName === 'error') throw new Error(data.message || '回答中断，请稍后重试。')
            onEvent(eventName, data)
            if (eventName === 'done') return data
          }
          eventName = 'message'
          dataLines = []
        } else if (line.startsWith('event:')) {
          eventName = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).replace(/^ /, ''))
        }
      }
      if (done) throw new Error('连接已中断，回答尚未完成，请重新提问。')
    }
  } catch (error) {
    if (controller.signal.aborted) throw new Error('回答超时，请稍后重试。')
    throw error
  } finally {
    clearTimeout(timer)
    if (reader) {
      try { await reader.cancel() } catch { /* 连接可能已经关闭 */ }
      reader.releaseLock()
    }
  }
}
