import { Service } from 'typedi';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import OpenAI from 'openai';
import { MemorySaver } from '@langchain/langgraph';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { WebSocket } from 'ws';
import {
  GoogleGenerativeAI,
  GenerativeModel,
  ChatSession as GeminiChatSession,
  HarmCategory,
  HarmBlockThreshold,
  SafetySetting,
  GenerationConfig,
} from '@google/generative-ai';

// Enhanced system prompt
const SYSTEM_PROMPT =
  'You are Lana, an intelligent and friendly AI assistant developed by Atlantis Software Inc. for the Atlantis AI product. Your goal is to provide clear, concise, and accurate responses in a user-friendly format. When responding, always give a brief summary that covers only the essential details—even if the query is detailed. If needed, ask if the user would like more information. Respond naturally and clearly, without unnecessary elaboration.\n\n**Response Formatting Guidelines:**\n- **Keep it concise:** Provide only the key details in a brief summary.\n- **Bold Key Information:** Wrap important details in double asterisks (e.g., **Temperature: 26.8°C**).\n- **Code:** Use triple backticks (```) with a preceding key (e.g., `code:`) for code snippets.\n- **Links:** Precede links with a key label (e.g., `link:`) to help differentiate them in the UI.\n\nAlways maintain a warm, professional tone and be proactive in clarifying ambiguities.';

// Gemini generation configuration
const GEMINI_CONFIG: GenerationConfig = {
  temperature: 0.9,
  topK: 40,
  topP: 0.8,
  maxOutputTokens: 4096, // Adjust if responses are long
  candidateCount: 1,
  stopSequences: ['User:', 'Assistant:'],
};

const SAFETY_SETTINGS: SafetySetting[] = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
  },
];

interface ChatHistory {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

interface ChatSession {
  chat: GeminiChatSession;
  history: ChatHistory[];
}

interface ConversationSession extends ChatSession {
  sessionId: string;
  userId?: string;
  createdAt: Date;
  lastUpdated: Date;
  context: {
    topic?: string;
    searchHistory: Array<{
      query: string;
      results: any[];
      timestamp: Date;
    }>;
    messageCount: number;
    lastInteraction: Date;
    summary?: string;
  };
}

@Service()
export class AgentService {
  private readonly agent;
  private readonly getWebSearchResults: TavilySearchResults;
  private readonly agentCheckpointer: MemorySaver;
  private readonly openai: OpenAI;
  private readonly systemPrompt: string;
  private readonly geminiModel: GenerativeModel;
  private readonly geminiChats: Map<string, ConversationSession>;
  private readonly searchCache: Map<string, { results: any[]; timestamp: number }> = new Map();
  private readonly SESSION_EXPIRY = 1000 * 60 * 60; // 1 hour

  constructor() {
    this.getWebSearchResults = new TavilySearchResults({
      maxResults: 3,
      apiKey: process.env.TAVILY_API_KEY,
      includeImages: true,
    });

    this.openai = new OpenAI({
      baseURL: 'https://openrouter.ai/api/v1',
      apiKey: process.env.OPENROUTER_API_KEY,
      defaultHeaders: {
        'HTTP-Referer': process.env.SITE_URL || 'http://localhost:3000',
        'X-Title': process.env.SITE_NAME || 'Local Development',
      },
    });

    const agentModel = new ChatOpenAI({
      modelName: 'openai/gpt-3.5-turbo',
      temperature: 0.7,
      streaming: true,
      configuration: {
        baseURL: 'https://openrouter.ai/api/v1',
        apiKey: process.env.OPENROUTER_API_KEY,
        defaultHeaders: {
          'HTTP-Referer': process.env.SITE_URL || 'http://localhost:3000',
          'X-Title': process.env.SITE_NAME || 'Local Development',
        },
      },
    });

    this.agentCheckpointer = new MemorySaver();
    this.agent = createReactAgent({
      llm: agentModel,
      tools: [this.getWebSearchResults],
      checkpointSaver: this.agentCheckpointer,
    });

    this.systemPrompt = SYSTEM_PROMPT;

    const apiKey = process.env.GOOGLE_GENAI_API_KEY;
    if (!apiKey) {
      throw new Error('GOOGLE_GENAI_API_KEY is not set in environment variables');
    }
    const genAI = new GoogleGenerativeAI(apiKey);
    this.geminiModel = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash',
      generationConfig: GEMINI_CONFIG,
      safetySettings: SAFETY_SETTINGS as unknown as SafetySetting[],
    });
    this.geminiChats = new Map();
    setInterval(() => this.cleanupSessions(), this.SESSION_EXPIRY);
  }

  /** Clean up inactive sessions */
  private cleanupSessions() {
    const now = Date.now();
    for (const [threadId, session] of this.geminiChats.entries()) {
      if (now - session.lastUpdated.getTime() > this.SESSION_EXPIRY) {
        this.geminiChats.delete(threadId);
      }
    }
  }

  /** Query using the react agent (non-streaming) */
  public async query(message: string, threadId: string) {
    try {
      const response = await this.agent.invoke(
        {
          messages: [
            { role: 'system', content: this.systemPrompt },
            { role: 'user', content: message },
          ],
        },
        { configurable: { thread_id: threadId } },
      );
      console.log(response);
      return response.messages[response.messages.length - 1].content;
    } catch (error) {
      console.error('Error in query:', error);
      throw error;
    }
  }

  /** Stream query using the react agent */
  public async streamQuery(message: string, threadId: string | undefined, ws: WebSocket): Promise<void> {
    if (!message || typeof message !== 'string') {
      return ws.send(JSON.stringify({ type: 'error', content: 'Invalid message', threadId }));
    }
    let streamedContent = '';
    try {
      await this.agent.invoke(
        {
          messages: [
            { role: 'system', content: this.systemPrompt },
            { role: 'user', content: message.trim() },
          ],
        },
        {
          configurable: { thread_id: threadId },
          callbacks: [
            {
              handleLLMNewToken: (token: string) => {
                if (token) {
                streamedContent += token;
                  ws.send(JSON.stringify({ role: 'assistant', type: 'token', content: token, threadId }));
                }
              },
              handleToolEnd: (output: string) => {
                if (output) {
                  ws.send(JSON.stringify({ role: 'assistant', type: 'tool', content: output, threadId }));
                }
              },
            },
          ],
        },
      );
      ws.send(JSON.stringify({ type: 'done', role: 'assistant', content: streamedContent, threadId }));
    } catch (error) {
      console.error('Error in streamQuery:', error);
      ws.send(JSON.stringify({ type: 'error', content: error instanceof Error ? error.message : 'Error processing query', threadId }));
    }
  }

  /** Direct OpenAI stream completion */
  public async streamDirectCompletion(message: string, ws: WebSocket, threadId?: string): Promise<void> {
    if (!message || typeof message !== 'string') {
      return ws.send(JSON.stringify({ type: 'error', content: 'Invalid message', threadId }));
    }
    let streamedContent = '';
    try {
      const stream = await this.openai.chat.completions.create({
        model: 'deepseek/deepseek-r1-distill-llama-70b:free',
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: message.trim() },
        ],
        stream: true,
      });
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          streamedContent += content;
          ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'token', content }));
        }
      }
      ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: streamedContent }));
    } catch (error) {
      console.error('Error in direct completion:', error);
      ws.send(JSON.stringify({ type: 'error', content: error instanceof Error ? error.message : 'Error processing query', threadId }));
    }
  }

  /** Gemini completion with context awareness and search integration */
  public async geminiCompletion(message: string, ws: WebSocket, threadId: string, userId?: string) {
    if (!message?.trim()) {
      return ws.send(JSON.stringify({ type: 'error', content: 'Invalid message', threadId }));
    }
    const trimmedMessage = message.trim();
    try {
      let session: ConversationSession;
      if (this.geminiChats.has(threadId)) {
        session = this.geminiChats.get(threadId)!;
        session.lastUpdated = new Date();
      } else {
        session = await this.initializeNewChat([], userId);
        this.geminiChats.set(threadId, session);
      }

      // Analyze whether the query needs fresh information.
      const analysisPrompt = `Question/Request: "${trimmedMessage}"

Please analyze if this query requires current or real-time information or fact-checking. If the request is for creative writing, summarization, or relies on general knowledge, respond with DIRECT.
Only respond with SEARCH if the query explicitly demands up-to-date factual information.
Respond in this format:
DECISION: [SEARCH or DIRECT]
REASON: [One brief sentence explaining why]`;

      const analysisResult = await session.chat.sendMessage(analysisPrompt);
      const analysisResponse = await analysisResult.response.text();
      const needsSearch = analysisResponse.includes('DECISION: SEARCH');
      const reason = analysisResponse.match(/REASON: (.*)/)?.[1] || '';

      if (needsSearch) {
        ws.send(
          JSON.stringify({
            role: 'assistant',
            threadId,
            type: 'tool',
            content: `I am looking for updated information about this topic 😁: ${reason}`,
          }),
        );
        await this.handleSearchAndResponse(session.chat, session, trimmedMessage, ws, threadId);
      } else {
        await this.handleNormalChat(session.chat, session.history, trimmedMessage, ws, threadId);
      }
    } catch (chatError) {
      console.error('Error in geminiCompletion:', chatError);
      ws.send(
        JSON.stringify({
          type: 'error',
          role: 'assistant',
          content: chatError instanceof Error ? chatError.message : 'Error processing query',
          threadId,
        }),
      );
      this.geminiChats.delete(threadId);
    }
  }

  /** Initialize a new Gemini chat session */
  private async initializeNewChat(history: ChatHistory[], userId?: string): Promise<ConversationSession> {
    const chat = this.geminiModel.startChat({
      history: [],
      generationConfig: GEMINI_CONFIG,
      safetySettings: SAFETY_SETTINGS,
    });
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    try {
      const initResult = await chat.sendMessage(SYSTEM_PROMPT);
      const initResponse = await initResult.response.text();
      return {
        sessionId,
        userId,
        chat,
        history: [
          { role: 'user', parts: [{ text: SYSTEM_PROMPT }] },
          { role: 'model', parts: [{ text: initResponse }] },
        ],
        createdAt: new Date(),
        lastUpdated: new Date(),
        context: {
          searchHistory: [],
          messageCount: 1,
          lastInteraction: new Date(),
        },
      };
    } catch (error) {
      console.error('Error initializing chat:', error);
      throw new Error('Failed to initialize chat session');
    }
  }

  /** Handle query that requires web search */
  private async handleSearchAndResponse(
    chat: GeminiChatSession,
    session: ConversationSession,
    message: string,
    ws: WebSocket,
    threadId: string,
  ): Promise<void> {
    let searchResults: any[];
    const cacheKey = message;
    const cacheEntry = this.searchCache.get(cacheKey);
    if (cacheEntry && Date.now() - cacheEntry.timestamp < 1000 * 60 * 5) {
      // 5 minutes cache
      searchResults = cacheEntry.results;
    } else {
      searchResults = await this.getWebSearchResults.invoke(message);
      this.searchCache.set(cacheKey, { results: searchResults, timestamp: Date.now() });
    }
    session.context.searchHistory.push({
      query: message,
      results: searchResults,
      timestamp: new Date(),
    });
    session.lastUpdated = new Date();

    // Build prompt including search results and ask for a brief, natural response.
    const messageWithContext = `User Query: "${message}"

Search Results: ${JSON.stringify(searchResults, null, 2)}

Previous Context: ${session.context.topic ? `This conversation is about ${session.context.topic}.` : 'No previous topic.'}
Search History: ${session.context.searchHistory.length} searches so far.

Task: Provide a **brief and natural response** that summarizes the key information without excessive detail. Cite sources as [URL] if applicable.`;

    await this.processAndStreamResponse(chat, session.history, messageWithContext, ws, threadId);

    if (!session.context.topic) {
      const topicAnalysis = await chat.sendMessage("Based on our conversation so far, what's the main topic? Respond with just 2-3 words.");
      session.context.topic = await topicAnalysis.response.text();
    }
  }

  /** Handle normal chat without additional search context */
  private async handleNormalChat(chat: GeminiChatSession, history: ChatHistory[], message: string, ws: WebSocket, threadId: string): Promise<void> {
    if (!this.geminiChats.has(threadId)) {
      const session = await this.initializeNewChat(history);
      this.geminiChats.set(threadId, session);
      chat = session.chat;
      history = session.history;
    } else {
      const existingSession = this.geminiChats.get(threadId)!;
      existingSession.lastUpdated = new Date();
    }
    const topicInfo = this.geminiChats.get(threadId)?.context.topic
      ? `This conversation is about ${this.geminiChats.get(threadId)?.context.topic}.`
      : 'No specific topic identified yet.';
    const messageWithContext = `User Query: "${message}"

Previous Context: ${topicInfo}

Task: Provide a **clear and natural response** with only the essential details, ensuring it is easy to read.`;
    await this.processAndStreamResponse(chat, history, messageWithContext, ws, threadId);

    const session = this.geminiChats.get(threadId)!;
    if (!session.context.topic) {
      const topicAnalysis = await chat.sendMessage("Based on our conversation so far, what's the main topic? Respond with just 2-3 words.");
      session.context.topic = await topicAnalysis.response.text();
    }
  }

  /** Process message and stream response token-by-token */
  private async processAndStreamResponse(
    chat: GeminiChatSession,
    history: ChatHistory[],
    message: string,
    ws: WebSocket,
    threadId: string,
  ): Promise<void> {
    const session = this.geminiChats.get(threadId);
    if (!session) return;
    session.context.messageCount++;
    session.context.lastInteraction = new Date();
    session.lastUpdated = new Date();
    history.push({ role: 'user', parts: [{ text: message }] });
    try {
      let streamedContent = '';
      const result = await chat.sendMessageStream(message);
      for await (const chunk of result.stream) {
        const content = chunk.text();
        if (content) {
          streamedContent += content;
          ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'token', content }));
        }
      }
      if (streamedContent) {
        history.push({ role: 'model', parts: [{ text: streamedContent }] });
        ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: streamedContent }));
      }
    } catch (error) {
      console.error('Error in stream processing:', error);
      ws.send(
        JSON.stringify({
          type: 'error',
          role: 'assistant',
          content: error instanceof Error ? error.message : 'Error generating response',
          threadId,
        }),
      );
    }
  }

  /** Direct (non-streaming) completion using OpenAI */
  public async directCompletion(message: string) {
    try {
      const completion = await this.openai.chat.completions.create({
        model: 'deepseek/deepseek-r1-distill-llama-70b:free',
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: message },
        ],
      });
      console.log(completion);
      return completion.choices[0]?.message.content;
    } catch (error) {
      console.error('Error in direct completion:', error);
      throw error;
    }
  }

  /** Retrieve session info */
  public async getSessionInfo(threadId: string): Promise<ConversationSession | null> {
    return this.geminiChats.get(threadId) || null;
  }

  /** Clear a session */
  public async clearSession(threadId: string): Promise<void> {
    this.geminiChats.delete(threadId);
  }
}
