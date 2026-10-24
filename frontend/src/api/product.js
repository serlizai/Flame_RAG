// 获取全部商品
export async function getProducts() {
  const response = await fetch('/api/products')

  if (!response.ok) {
    throw new Error(`获取商品列表失败：HTTP ${response.status}`)
  }

  return response.json()
}

// 根据商品 ID 获取商品详情
export async function getProductById(productId) {
  const response = await fetch(
    `/api/products/${encodeURIComponent(productId)}`
  )

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('该商品不存在')
    }

    throw new Error(`获取商品详情失败：HTTP ${response.status}`)
  }

  return response.json()
}