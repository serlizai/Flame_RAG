
<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { getProducts, getProductById } from './api/product'
import { sendChatMessage } from './api/chat'

// -------------------- 商品数据 --------------------

const products = ref([])
// 商品搜索关键词
const searchKeyword = ref('')

// 当前选择的分类，all 表示全部
const selectedCategory = ref('all')

// 根据后端商品数据自动生成分类
const categories = computed(() => {
  const values = products.value
    .map(product => product.category)
    .filter(Boolean)

  return [...new Set(values)]
})

// 根据关键词和分类筛选商品
const filteredProducts = computed(() => {
  const keyword = searchKeyword.value.trim().toLowerCase()

  return products.value.filter(product => {
    const matchesCategory =
      selectedCategory.value === 'all' ||
      product.category === selectedCategory.value

    const matchesKeyword =
      !keyword ||
      product.name.toLowerCase().includes(keyword) ||
      product.id.toLowerCase().includes(keyword)

    return matchesCategory && matchesKeyword
  })
})

const selectedId = ref('')
const productsLoading = ref(true)
const productsError = ref('')
const scanError = ref('')

// 从二维码网址中读取商品编号
const params = new URLSearchParams(window.location.search)
const scannedProductId = params.get('product_id')

// 当前商品
const currentProduct = computed(() => {
  return products.value.find(
    product => product.id === selectedId.value
  ) ?? null
})

// 商品分类展示名称
const categoryNames = {
  smartphone: '智能手机',
  kitchen_appliance: '厨房电器',
  lighting: '照明设备',
  home_appliance: '家用电器',
  drinkware: '饮具',
  home_storage: '家居收纳',
  beverage: '饮料饮品'
}

// 商品分类图标（没有真实图片时作为占位）
const categoryIcons = {
  smartphone: '📱',
  kitchen_appliance: '🍳',
  lighting: '💡',
  home_appliance: '🏠',
  drinkware: '🥤',
  home_storage: '📦',
  beverage: '🥤'
}

function categoryLabel(category) {
  return categoryNames[category] || category || '精选商品'
}

function categoryIcon(category) {
  return categoryIcons[category] || '✨'
}

// -------------------- 加载商品 --------------------

async function loadProducts() {
  productsLoading.value = true
  productsError.value = ''
  scanError.value = ''

  try {
    const data = await getProducts()

    if (!Array.isArray(data)) {
      throw new Error('商品列表返回格式不正确')
    }

    products.value = data

    if (scannedProductId) {
      const exists = data.some(
        product => product.id === scannedProductId
      )

      if (exists) {
        selectedId.value = scannedProductId

        // 同时验证商品详情接口
        await getProductById(scannedProductId)
      } else {
        selectedId.value = ''
        scanError.value =
          '没有找到二维码对应的商品，请检查二维码或联系工作人员。'
      }
    } else if (data.length > 0) {
      selectedId.value = data[0].id
    }
  } catch (error) {
    productsError.value =
      error.message || '商品加载失败，请稍后重试'
  } finally {
    productsLoading.value = false
  }
}

onMounted(loadProducts)

// -------------------- 聊天 --------------------

const inputMessage = ref('')
const messages = ref([])
const isLoading = ref(false)
const sessionId = ref(null)
const chatContainer = ref(null)

const quickQuestions = [
  '这款商品有什么特点？',
  '有哪些主要参数？',
  '应该怎么使用？',
  '售后政策是什么？'
]

watch(selectedId, () => {
  messages.value = []
  sessionId.value = null
  inputMessage.value = ''
})

async function scrollToBottom() {
  await nextTick()

  const element = chatContainer.value

  if (element) {
    element.scrollTop = element.scrollHeight
  }
}

// 选择商品
function selectProduct(product) {
  if (isLoading.value) return

  selectedId.value = product.id
  scanError.value = ''

  // 把当前商品 ID 同步到网址中
  const url = new URL(window.location.href)
  url.searchParams.set('product_id', product.id)

  window.history.replaceState({}, '', url)

  scrollToBottom()
}

// 请求真实 AI 回答
async function sendMessage(text = inputMessage.value) {
  const question = String(text).trim()

  if (!question || isLoading.value || !currentProduct.value) {
    return
  }

  const productId = selectedId.value

  messages.value.push({
    role: 'user',
    content: question
  })

  inputMessage.value = ''
  isLoading.value = true

  await scrollToBottom()

  try {
    const result = await sendChatMessage({
      productId,
      message: question,
      sessionId: sessionId.value
    })

    // 保存后端会话编号，支持多轮聊天
    sessionId.value = result.session_id

    messages.value.push({
      role: 'assistant',
      content: result.answer || '暂未获得有效回答。',
      citations: Array.isArray(result.citations)
        ? result.citations
        : []
    })
  } catch (error) {
    messages.value.push({
      role: 'error',
      content:
        'AI 服务暂时无法完成回答，请检查后端模型、知识库和 Redis 是否已正常配置。'
    })

    console.error('AI 问答失败：', error)
  } finally {
    isLoading.value = false
    await scrollToBottom()
  }
}
</script>

<template>
  <div class="page">
    <!-- 网站顶部 -->
    <header class="navbar">
      <div class="logo">
        <div class="logo-symbol">✦</div>
        <span>FLAME<span class="logo-accent">AI</span></span>
      </div>

      <span class="navbar-subtitle">
        探索每一件好物的更多可能
      </span>

      <div class="navbar-label">
        <span class="status-dot"></span>
        智能商品探索
      </div>
    </header>

    <!-- 主页面 -->
    <main class="main-layout">

      <!-- 左侧：商品展示 -->
      <section class="product-section">

        <div class="intro">
          <span class="eyebrow">
            ✦ AI SHOPPING EXPERIENCE
          </span>

          <h1>
            每一次探索，<br />
            都有<span>新发现。</span>
          </h1>

          <p>
            从一件好物开始，让 AI 帮你了解产品、
            探索特点，找到适合自己的选择。
          </p>
        </div>

        <!-- 商品加载状态 -->
        <div v-if="productsLoading" class="state-card">
          <div class="state-icon">◌</div>
          <h3>正在发现好物...</h3>
          <p>正在从商品服务获取最新商品信息</p>
        </div>

        <!-- 接口失败 -->
        <div v-else-if="productsError" class="state-card">
          <div class="state-icon">!</div>
          <h3>商品暂时无法加载</h3>
          <p>{{ productsError }}</p>
          <button class="primary-button" @click="loadProducts">
            重新加载
          </button>
        </div>

        <!-- 错误二维码 -->
        <div v-else-if="scanError" class="state-card">
          <div class="state-icon">?</div>
          <h3>商品识别失败</h3>
          <p>{{ scanError }}</p>
          <button
            class="primary-button"
            @click="selectProduct(products[0])"
            :disabled="products.length === 0"
          >
            浏览商品列表
          </button>
        </div>

        <template v-else-if="currentProduct">
          <!-- 大型商品视觉区 -->
          <div class="product-stage">
            <div class="stage-circle circle-one"></div>
            <div class="stage-circle circle-two"></div>

            <span class="floating-star star-one">✦</span>
            <span class="floating-star star-two">✳</span>

            <div class="product-icon">
              {{ categoryIcon(currentProduct.category) }}
            </div>

            <span class="stage-caption">
              DISCOVER YOUR PRODUCT
            </span>
          </div>

          <!-- 当前商品卡片 -->
          <div class="featured-card">
            <div class="featured-top">
              <span class="small-title">当前探索商品</span>
              <span class="category-badge">
                {{ categoryLabel(currentProduct.category) }}
              </span>
            </div>

            <h2>{{ currentProduct.name }}</h2>

            <p class="product-subtitle">
              发现商品的更多信息，让 AI 为你详细解答。
            </p>

            <div class="product-meta">
              <div>
                <span>商品编号</span>
                <strong>{{ currentProduct.id }}</strong>
              </div>

            </div>

            <button
              class="explore-button"
              @click="sendMessage('请介绍一下这款商品')"
              :disabled="isLoading"
            >
              ✨ 让 AI 介绍这款商品
              <span>↗</span>
            </button>
          </div>
        </template>

        <div v-else class="state-card">
          <h3>暂无商品</h3>
          <p>商品库目前还没有可以展示的商品。</p>
        </div>

        <!-- 所有商品 -->
        <div class="catalog-section">
          <div class="section-title">
            <div>
              <span class="eyebrow">EXPLORE MORE</span>
              <h2>发现更多好物</h2>
            </div>

            <span class="total-count">
              {{ products.length }} 款商品
            </span>
          </div>

          <!-- 商品搜索 -->
          <div class="search-box">
            <span>🔍</span>

            <input
              v-model="searchKeyword"
              type="text"
              placeholder="搜索商品名称..."
            />

            <button
              v-if="searchKeyword"
              type="button"
              @click="searchKeyword = ''"
              aria-label="清空搜索"
            >
              ×
            </button>
          </div>

          <!-- 分类筛选 -->
          <div class="category-filters">
            <button
              type="button"
              :class="{ active: selectedCategory === 'all' }"
              @click="selectedCategory = 'all'"
            >
              全部商品
            </button>

            <button
              v-for="category in categories"
              :key="category"
              type="button"
              :class="{ active: selectedCategory === category }"
              @click="selectedCategory = category"
            >
              {{ categoryLabel(category) }}
            </button>
          </div>

          <div v-if="products.length > 0" class="product-grid">
            <button
              v-for="product in filteredProducts"
              :key="product.id"
              type="button"
              class="product-tile"
              :class="{
                active: selectedId === product.id
              }"
              :disabled="isLoading"
              @click="selectProduct(product)"
            >
              <div class="tile-image">
                {{ categoryIcon(product.category) }}
              </div>

              <div class="tile-content">
                <span>{{ categoryLabel(product.category) }}</span>
                <strong>{{ product.name }}</strong>
              </div>

              <span class="tile-arrow">↗</span>
            </button>
          </div>

          <div
            v-if="products.length > 0 && filteredProducts.length === 0"
            class="no-search-results"
          >
            没有找到符合条件的商品，试试其他关键词吧。
          </div>

          <div v-if="products.length === 0" class="empty-catalog">
            当前暂无可展示商品
          </div>
        </div>
      </section>

      <!-- 右侧：AI 导购 -->
      <section class="assistant-section">
        <div class="assistant-title">
          <div class="assistant-title-icon">✦</div>

          <div>
            <span class="eyebrow">YOUR AI ASSISTANT</span>
            <h2>嗨，今天想了解什么？</h2>
            <p>和 AI 一起，轻松探索商品信息。</p>
          </div>
        </div>

        <div class="chat-card">
          <!-- 聊天顶部 -->
          <div class="chat-header">
            <div class="chat-header-left">
              <div class="chat-avatar">✦</div>

              <div>
                <strong>Flame 智能导购</strong>
                <span>AI 商品咨询助手</span>
              </div>
            </div>

            <span class="chat-tag">AI POWERED</span>
          </div>

          <!-- 聊天记录 -->
          <div ref="chatContainer" class="chat-body">
            <div class="chat-welcome">
              <div class="welcome-icon">✳</div>
              <h3>你好，很高兴认识你！</h3>

              <p v-if="currentProduct">
                我们现在正在探索<br />
                <strong>{{ currentProduct.name }}</strong>
              </p>

              <p v-else>
                请选择一件商品，开始你的探索。
              </p>

              <span>你可以直接提问，也可以点击快捷问题</span>
            </div>

            <div
              v-for="(message, index) in messages"
              :key="index"
              class="message-row"
              :class="message.role"
            >
              <div
                v-if="message.role !== 'user'"
                class="message-avatar"
              >
                ✦
              </div>

              <div class="message-content">
                <div class="message-bubble">
                  {{ message.content }}
                </div>

                <!-- 回答引用资料 -->
                <div
                  v-if="message.citations?.length"
                  class="citation-list"
                >
                  <details
                    v-for="(citation, citationIndex) in message.citations"
                    :key="citationIndex"
                    class="citation-item"
                  >
                    <summary>
                      📄 参考资料
                      {{ citation.number ?? citationIndex + 1 }}
                      ·
                      {{ citation.label || citation.source || '商品资料' }}
                    </summary>

                    <p>
                      {{ citation.snippet || '暂无引用摘要' }}
                    </p>
                  </details>
                </div>
              </div>
            </div>

            <div v-if="isLoading" class="message-row assistant">
              <div class="message-avatar">✦</div>
              <div class="loading-message">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>

          <!-- 聊天输入区域 -->
          <div class="chat-bottom">
            <div class="quick-header">
              <span>✧ 猜你想问</span>
              <span>点击即可提问</span>
            </div>

            <div class="quick-questions">
              <button
                v-for="question in quickQuestions"
                :key="question"
                type="button"
                :disabled="isLoading || !currentProduct"
                @click="sendMessage(question)"
              >
                {{ question }}
                <span>↗</span>
              </button>
            </div>

            <form
              class="input-box"
              @submit.prevent="sendMessage()"
            >
              <input
                v-model="inputMessage"
                type="text"
                placeholder="问问 AI 关于这款商品..."
                :disabled="isLoading || !currentProduct"
              />

              <button
                type="submit"
                :disabled="
                  !inputMessage.trim() ||
                  isLoading ||
                  !currentProduct
                "
                aria-label="发送消息"
              >
                ↑
              </button>
            </form>

            <p class="chat-note">
              AI 回答仅供参考，具体信息以商品正式资料为准。
            </p>
          </div>
        </div>

        <div class="feature-grid">
          <div class="feature-card">
            <span>🔍</span>
            <strong>商品探索</strong>
            <p>了解商品信息</p>
          </div>

          <div class="feature-card">
            <span>✨</span>
            <strong>智能问答</strong>
            <p>AI 解答疑问</p>
          </div>

          <div class="feature-card">
            <span>🎁</span>
            <strong>互动福利</strong>
            <p>后续开放</p>
          </div>
        </div>
      </section>
    </main>

    <footer class="footer">
      <span>✦ FLAME AI</span>
      <span>让每一次探索都有新发现</span>
    </footer>
  </div>
</template>

<style scoped>
.page {
  --orange: #ff6b35;
  --orange-light: #fff0e8;
  --ink: #242424;
  --muted: #89837e;
  --cream: #fff5ed;
  --border: #ebe7e2;

  width: 100%;
  min-height: 100dvh;
  background: var(--cream);
  color: var(--ink);
  overflow-x: hidden;
}

/* 顶部导航 */
.navbar {
  height: 76px;
  background: #fff;
  border-bottom: 1px solid var(--border);
  padding: 0 5%;

  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 23px;
  font-weight: 900;
}

.logo-symbol,
.logo-accent {
  color: var(--orange);
}

.logo-symbol {
  font-size: 30px;
}

.navbar-subtitle {
  font-size: 12px;
  color: var(--muted);
}

.navbar-label {
  font-size: 11px;
  color: #666;
}

.status-dot {
  display: inline-block;
  height: 8px;
  width: 8px;
  border-radius: 50%;
  margin-right: 6px;
  background: #55b891;
}

/* 页面主体 */
.main-layout {
  max-width: 1500px;
  margin: 0 auto;
  padding: 42px 5% 65px;

  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 42px;
}

.product-section,
.assistant-section {
  min-width: 0;
}

.eyebrow {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.5px;
  color: var(--orange);
}

.intro h1 {
  font-size: clamp(39px, 4.5vw, 68px);
  line-height: 1.16;
  letter-spacing: -2px;
  margin: 18px 0;
}

.intro h1 span {
  color: var(--orange);
}

.intro p {
  max-width: 480px;
  font-size: 13px;
  color: var(--muted);
  line-height: 2;
}

/* 商品视觉区 */
.product-stage {
  height: 310px;
  margin-top: 24px;
  margin-bottom: 22px;

  background: radial-gradient(
    circle at center,
    #ffe0b9,
    #ffecdc 58%,
    #fff2e7
  );

  position: relative;
  overflow: hidden;
  border-radius: 28px;

  display: flex;
  align-items: center;
  justify-content: center;
}

.stage-circle {
  position: absolute;
  border: 1px solid rgba(255, 255, 255, 0.85);
  border-radius: 50%;
}

.circle-one {
  width: 240px;
  height: 240px;
}

.circle-two {
  width: 360px;
  height: 360px;
}

.product-icon {
  z-index: 1;
  font-size: 115px;
  filter: drop-shadow(0 20px 16px rgba(120, 72, 35, 0.15));
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-11px); }
}

.floating-star {
  position: absolute;
  color: var(--orange);
  font-size: 28px;
}

.star-one {
  top: 16%;
  left: 17%;
}

.star-two {
  bottom: 18%;
  right: 16%;
}

.stage-caption {
  position: absolute;
  bottom: 19px;
  font-size: 10px;
  letter-spacing: 2px;
  color: #bc9478;
}

/* 当前商品 */
.featured-card,
.chat-card {
  background: white;
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: 0 15px 50px rgba(40, 30, 20, 0.04);
}

.featured-card {
  padding: 25px;
}

.featured-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.small-title {
  font-size: 12px;
  color: var(--muted);
}

.category-badge {
  font-size: 10px;
  color: var(--orange);
  border-radius: 100px;
  padding: 7px 12px;
  background: var(--orange-light);
}

.featured-card h2 {
  font-size: 27px;
  margin: 20px 0 8px;
  overflow-wrap: anywhere;
}

.product-subtitle {
  font-size: 12px;
  color: var(--muted);
}

.product-meta {
  display: flex;
  gap: 20px;
  margin: 23px 0;
  padding-top: 18px;
  border-top: 1px solid var(--border);
}

.product-meta div {
  min-width: 0;
  flex: 1;
}

.product-meta span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 6px;
}

.product-meta strong {
  font-size: 12px;
  overflow-wrap: anywhere;
}

.explore-button,
.primary-button {
  background: var(--orange);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 14px 18px;
  font-size: 13px;
  font-weight: 700;
}

.explore-button {
  width: 100%;
  display: flex;
  justify-content: space-between;
}

.explore-button:disabled,
.primary-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.state-card {
  min-height: 290px;
  margin-top: 25px;
  padding: 35px 20px;
  background: white;
  border: 1px solid var(--border);
  border-radius: 22px;

  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.state-icon {
  font-size: 40px;
  color: var(--orange);
}

.state-card h3 {
  font-size: 19px;
}

.state-card p {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.8;
}

/* 全部商品 */
.catalog-section {
  margin-top: 34px;
}

.section-title {
  display: flex;
  justify-content: space-between;
  align-items: end;
  margin-bottom: 19px;
}

.section-title h2 {
  font-size: 23px;
  margin: 8px 0 0;
}

.total-count {
  font-size: 11px;
  color: var(--muted);
}

.search-box {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #fff;
  border: 1px solid #ebe7e2;
  border-radius: 13px;
  padding: 12px 15px;
  margin-bottom: 14px;
}

.search-box > span {
  flex-shrink: 0;
  font-size: 14px;
}

.search-box input {
  flex: 1;
  min-width: 0;
  border: none;
  outline: none;
  background: transparent;
  font-size: 13px;
}

.search-box input::placeholder {
  color: var(--muted);
}

.search-box:focus-within {
  border-color: var(--orange);
}

.search-box button {
  border: none;
  background: none;
  color: #89837e;
  font-size: 18px;
}

.category-filters {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 10px;
  margin-bottom: 12px;
}

.category-filters button {
  flex-shrink: 0;
  border: 1px solid #ebe7e2;
  background: white;
  color: #777;
  padding: 9px 13px;
  border-radius: 100px;
  font-size: 11px;
}

.category-filters button:hover,
.category-filters button.active {
  background: #ff6b35;
  border-color: #ff6b35;
  color: white;
}

.product-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.no-search-results {
  padding: 28px 12px;
  text-align: center;
  color: #89837e;
  font-size: 12px;
}

.product-tile {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  text-align: left;
  padding: 13px;

  background: white;
  border: 1px solid var(--border);
  border-radius: 15px;
  transition: 0.2s;
}

.product-tile:hover {
  border-color: var(--orange);
  transform: translateY(-2px);
}

.product-tile.active {
  border: 1.5px solid var(--orange);
  background: #fff8f3;
}

.product-tile:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.tile-image {
  width: 54px;
  height: 54px;
  flex-shrink: 0;
  border-radius: 12px;
  background: var(--orange-light);

  display: grid;
  place-items: center;
  font-size: 25px;
}

.tile-content {
  min-width: 0;
  flex: 1;
}

.tile-content span {
  display: block;
  color: var(--muted);
  font-size: 10px;
  margin-bottom: 6px;
}

.tile-content strong {
  font-size: 12px;
  line-height: 1.4;
  display: block;
  overflow-wrap: anywhere;
}

.tile-arrow {
  font-size: 15px;
  color: var(--orange);
}

.empty-catalog {
  color: var(--muted);
  padding: 20px;
}

/* AI 助手标题 */
.assistant-title {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 22px;
}

.assistant-title-icon {
  width: 56px;
  height: 56px;
  border-radius: 18px;
  background: var(--orange);
  color: white;

  display: grid;
  place-items: center;
  font-size: 30px;
  flex-shrink: 0;
}

.assistant-title h2 {
  font-size: 23px;
  margin: 7px 0;
}

.assistant-title p {
  font-size: 12px;
  color: var(--muted);
  margin: 0;
}

/* 聊天 */
.chat-card {
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 590px;
}

.chat-header {
  padding: 18px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.chat-header-left {
  display: flex;
  align-items: center;
  gap: 11px;
}

.chat-avatar,
.message-avatar {
  flex-shrink: 0;
  background: var(--orange);
  color: white;
  display: grid;
  place-items: center;
}

.chat-avatar {
  width: 42px;
  height: 42px;
  border-radius: 13px;
  font-size: 25px;
}

.chat-header-left strong,
.chat-header-left span {
  display: block;
}

.chat-header-left strong {
  font-size: 13px;
}

.chat-header-left span {
  font-size: 11px;
  color: var(--muted);
  margin-top: 5px;
}

.chat-tag {
  background: var(--orange-light);
  color: var(--orange);
  padding: 8px 10px;
  border-radius: 30px;
  font-size: 10px;
}

.chat-body {
  flex: 1;
  height: 330px;
  min-height: 330px;
  max-height: 520px;
  overflow-y: auto;
  padding: 20px;
}

.chat-welcome {
  text-align: center;
  padding: 15px 0 30px;
}

.welcome-icon {
  font-size: 43px;
  color: var(--orange);
}

.chat-welcome h3 {
  font-size: 18px;
  margin: 8px 0;
}

.chat-welcome p {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.9;
}

.chat-welcome strong {
  color: var(--ink);
}

.chat-welcome > span {
  font-size: 10px;
  color: #b0a9a3;
}

.message-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 17px;
}

.message-row.user {
  justify-content: flex-end;
}

.message-avatar {
  width: 27px;
  height: 27px;
  border-radius: 9px;
}

.message-content {
  max-width: 85%;
  min-width: 0;
}

.message-bubble {
  background: #f5f4f2;
  padding: 13px;
  border-radius: 4px 14px 14px 14px;
  font-size: 12px;
  line-height: 1.8;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.user .message-bubble {
  background: var(--orange);
  color: white;
  border-radius: 14px 4px 14px 14px;
}

.error .message-bubble {
  background: #fff1ed;
  color: #a44325;
}

.citation-list {
  margin-top: 8px;
}

.citation-item {
  margin-top: 6px;
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 9px;
}

.citation-item summary {
  cursor: pointer;
  color: var(--orange);
  font-size: 11px;
  overflow-wrap: anywhere;
}

.citation-item p {
  color: var(--muted);
  line-height: 1.7;
  font-size: 11px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.loading-message {
  display: flex;
  gap: 6px;
  padding: 12px;
}

.loading-message span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--orange);
  animation: blink 0.8s infinite alternate;
}

.loading-message span:nth-child(2) {
  animation-delay: 0.2s;
}

.loading-message span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes blink {
  to {
    opacity: 0.25;
    transform: translateY(-3px);
  }
}

/* 聊天底部 */
.chat-bottom {
  border-top: 1px solid var(--border);
  padding: 18px;
}

.quick-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 11px;
}

.quick-header span:first-child {
  font-size: 12px;
  font-weight: 700;
}

.quick-header span:last-child {
  font-size: 10px;
  color: var(--muted);
}

.quick-questions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 18px;
}

.quick-questions button {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  text-align: left;
  border: 1px solid var(--border);
  background: #faf9f6;
  border-radius: 10px;
  padding: 11px;
  font-size: 11px;
}

.quick-questions button span {
  color: var(--orange);
}

.quick-questions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.input-box {
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid #e7e2dd;
  border-radius: 13px;
  padding: 7px;
}

.input-box:focus-within {
  border-color: var(--orange);
}

.input-box input {
  min-width: 0;
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  padding-left: 9px;
  font-size: 12px;
}

.input-box button {
  width: 38px;
  height: 38px;
  flex-shrink: 0;
  border: none;
  border-radius: 10px;
  background: var(--orange);
  color: white;
  font-size: 23px;
}

.input-box button:disabled {
  opacity: 0.5;
}

.chat-note {
  text-align: center;
  font-size: 10px;
  color: var(--muted);
  margin: 12px 0 0;
}

/* 功能卡片 */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 16px;
}

.feature-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 8px;
  background: white;
  text-align: center;
}

.feature-card > span {
  display: block;
  font-size: 24px;
  margin-bottom: 8px;
}

.feature-card strong {
  font-size: 12px;
}

.feature-card p {
  font-size: 10px;
  color: var(--muted);
}

/* 底部 */
.footer {
  border-top: 1px solid var(--border);
  padding: 22px 5%;
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-size: 11px;
}

/* 平板和手机 */
@media (max-width: 900px) {
  .main-layout {
    grid-template-columns: 1fr;
    padding: 25px 18px 45px;
    gap: 34px;
  }

  .navbar {
    height: 65px;
    padding: 0 18px;
  }

  .navbar-label {
    display: none;
  }

  .intro h1 {
    font-size: 43px;
  }

  .product-stage {
    height: 265px;
  }

  .product-icon {
    font-size: 100px;
  }
}

@media (max-width: 420px) {
  .navbar-subtitle {
    display: none;
  }

  .intro h1 {
    font-size: 39px;
  }

  .featured-card {
    padding: 20px;
  }

  .product-grid {
    gap: 9px;
  }

  .product-tile {
    padding: 9px;
    gap: 7px;
  }

  .tile-image {
    width: 42px;
    height: 42px;
    font-size: 21px;
  }

  .tile-content strong {
    font-size: 11px;
  }

  .tile-arrow {
    display: none;
  }

  .assistant-title h2 {
    font-size: 20px;
  }

  .chat-header,
  .chat-bottom {
    padding: 13px;
  }

  .chat-body {
    padding: 13px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .product-icon,
  .loading-message span {
    animation: none;
  }
}
</style>
