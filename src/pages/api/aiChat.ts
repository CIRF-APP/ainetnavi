import { Message } from '@/features/messages/messages'
import { createOpenAI } from '@ai-sdk/openai'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { createCohere } from '@ai-sdk/cohere'
import { createMistral } from '@ai-sdk/mistral'
import { createAzure } from '@ai-sdk/azure'
import { streamText, generateText, CoreMessage } from 'ai'
import { NextRequest } from 'next/server'

type AIServiceKey =
  | 'openai'
  | 'anthropic'
  | 'google'
  | 'azure'
  | 'groq'
  | 'cohere'
  | 'mistralai'
  | 'perplexity'
  | 'fireworks'
type AIServiceConfig = Record<AIServiceKey, () => any>

// Allow streaming responses up to 30 seconds
export const maxDuration = 30

export const config = {
  runtime: 'edge',
}

export default async function handler(req: NextRequest) {
  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({
        error: 'Method Not Allowed',
        errorCode: 'METHOD_NOT_ALLOWED',
      }),
      {
        status: 405,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }

  const { messages, apiKey, aiService, model, stream } = await req.json()

  let aiApiKey = apiKey
  if (!aiApiKey) {
    const envKey = `${aiService.toUpperCase()}_KEY` as keyof typeof process.env
    const envApiKey = process.env[envKey]

    aiApiKey = envApiKey
  }

  if (!aiApiKey) {
    return new Response(
      JSON.stringify({ error: 'Empty API Key', errorCode: 'EmptyAPIKey' }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }

  if (!aiService || !model) {
    return new Response(
      JSON.stringify({
        error: 'Invalid AI service or model',
        errorCode: 'AIInvalidProperty',
      }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }

  const aiServiceConfig: AIServiceConfig = {
    openai: () => createOpenAI({ apiKey: aiApiKey }),
    anthropic: () => createAnthropic({ apiKey: aiApiKey }),
    google: () => createGoogleGenerativeAI({ apiKey: aiApiKey }),
    azure: () =>
      createAzure({
        resourceName:
          model.match(/https:\/\/(.+?)\.openai\.azure\.com/)?.[1] || '',
        apiKey: aiApiKey,
      }),
    groq: () =>
      createOpenAI({
        baseURL: 'https://api.groq.com/openai/v1',
        apiKey: aiApiKey,
      }),
    cohere: () => createCohere({ apiKey: aiApiKey }),
    mistralai: () => createMistral({ apiKey: aiApiKey }),
    perplexity: () =>
      createOpenAI({ baseURL: 'https://api.perplexity.ai/', apiKey: aiApiKey }),
    fireworks: () =>
      createOpenAI({
        baseURL: 'https://api.fireworks.ai/inference/v1',
        apiKey: aiApiKey,
      }),
  }
  const aiServiceInstance = aiServiceConfig[aiService as AIServiceKey]

  if (!aiServiceInstance) {
    return new Response(
      JSON.stringify({
        error: 'Invalid AI service',
        errorCode: 'InvalidAIService',
      }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }

  const instance = aiServiceInstance()
  const modifiedMessages: Message[] = modifyMessages(aiService, messages)
  let modifiedModel = model
  if (aiService === 'azure') {
    modifiedModel =
      model.match(/\/deployments\/(.+?)\/completions/)?.[1] || model
  }

  try {
    // Gemini APIの場合は直接API呼び出しを使用
    if (aiService === 'google') {
      return await handleGeminiAPI(
        aiApiKey,
        modifiedModel,
        modifiedMessages,
        stream
      )
    }

    // その他のAIサービスは既存の実装を使用
    if (stream) {
      const result = await streamText({
        model: instance(modifiedModel),
        messages: modifiedMessages as CoreMessage[],
      })

      return result.toTextStreamResponse()
    } else {
      const result = await generateText({
        model: instance(model),
        messages: modifiedMessages as CoreMessage[],
      })

      return new Response(JSON.stringify({ text: result.text }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }
  } catch (error) {
    console.error('Error in AI API call:', error)

    return new Response(
      JSON.stringify({
        error: 'Unexpected Error',
        errorCode: 'AIAPIError',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}

// Gemini API直接呼び出し用のハンドラー
async function handleGeminiAPI(
  apiKey: string,
  model: string,
  messages: Message[],
  stream: boolean
): Promise<Response> {
  const { systemInstruction, contents } = convertToGeminiFormat(messages)

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:${
    stream ? 'streamGenerateContent?alt=sse' : 'generateContent'
  }`

  const requestBody: any = { contents }
  if (systemInstruction) {
    requestBody.system_instruction = systemInstruction
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'x-goog-api-key': apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    console.error('Gemini API Error:', errorData)
    return new Response(
      JSON.stringify({
        error: 'Gemini API Error',
        errorCode: 'GeminiAPIError',
        details: errorData,
      }),
      {
        status: response.status,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }

  if (stream) {
    // ストリーミングレスポンスの処理
    return handleGeminiStreamResponse(response)
  } else {
    // 非ストリーミングレスポンスの処理
    const data = await response.json()
    const text =
      data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response'

    return new Response(JSON.stringify({ text }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}

// Geminiのストリーミングレスポンスを処理
async function handleGeminiStreamResponse(
  response: Response
): Promise<Response> {
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()

  const stream = new ReadableStream({
    async start(controller) {
      const reader = response.body?.getReader()
      if (!reader) {
        controller.close()
        return
      }

      let buffer = ''

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                if (line.slice(6).trim() === '') continue

                const jsonData = JSON.parse(line.slice(6))
                const text =
                  jsonData.candidates?.[0]?.content?.parts?.[0]?.text || ''

                if (text) {
                  // Vercel AI SDKと互換性のある形式で送信
                  controller.enqueue(
                    encoder.encode(`0:${JSON.stringify(text)}\n`)
                  )
                }
              } catch (e) {
                console.error('Parse error:', e, 'on line:', line)
              }
            }
          }
        }
      } catch (error) {
        console.error('Stream error:', error)
        controller.error(error)
      } finally {
        reader.releaseLock()
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Transfer-Encoding': 'chunked',
    },
  })
}

// メッセージをGemini API形式に変換
function convertToGeminiFormat(messages: Message[]): {
  systemInstruction: any | null
  contents: any[]
} {
  const systemPrompt = messages.find((msg) => msg.role === 'system')?.content

  const contents = messages
    .filter((message) => message.role !== 'system')
    .map((message) => ({
      role: message.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: message.content }],
    }))

  const systemInstruction = systemPrompt
    ? { parts: [{ text: systemPrompt }] }
    : null

  return { systemInstruction, contents }
}

function modifyMessages(aiService: string, messages: Message[]): Message[] {
  if (aiService === 'anthropic' || aiService === 'perplexity') {
    return modifyAnthropicMessages(messages)
  }
  return messages
}

// Anthropicのメッセージを修正する
function modifyAnthropicMessages(messages: Message[]): Message[] {
  const systemMessage: Message | undefined = messages.find(
    (message) => message.role === 'system'
  )
  let userMessages = messages
    .filter((message) => message.role !== 'system')
    .filter((message) => message.content !== '')

  userMessages = consolidateMessages(userMessages)

  while (userMessages.length > 0 && userMessages[0].role !== 'user') {
    userMessages.shift()
  }

  const result: Message[] = systemMessage
    ? [systemMessage, ...userMessages]
    : userMessages
  return result
}

// 同じroleのメッセージを結合する
function consolidateMessages(messages: Message[]) {
  const consolidated: Message[] = []
  let lastRole: string | null = null
  let combinedContent:
    | string
    | [
        {
          type: 'text'
          text: string
        },
        {
          type: 'image'
          image: string
        },
      ]

  messages.forEach((message, index) => {
    if (message.role === lastRole) {
      if (typeof combinedContent === 'string') {
        combinedContent += '\n' + message.content
      } else {
        combinedContent[0].text += '\n' + message.content
      }
    } else {
      if (lastRole !== null) {
        consolidated.push({ role: lastRole, content: combinedContent })
      }
      lastRole = message.role
      combinedContent = message.content
    }

    if (index === messages.length - 1) {
      consolidated.push({ role: lastRole, content: combinedContent })
    }
  })

  return consolidated
}
