import { ChatHistory, ConversationSession } from '@/interfaces/agent.interface';
import {
  ChatSession as GeminiChatSession,
  GenerationConfig,
  GenerativeModel,
  GoogleGenerativeAI,
  HarmBlockThreshold,
  HarmCategory,
  SafetySetting,
} from '@google/generative-ai';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { MemorySaver } from '@langchain/langgraph';
import { createReactAgent, toolsCondition } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import OpenAI from 'openai';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { Service } from 'typedi';
import { WebSocket } from 'ws';

// System prompt instructing Lana to answer directly and then include a follow-up question (without a visible prefix)
const SYSTEM_PROMPT =
  'You are Lana, an intelligent and friendly AI assistant developed by Atlantis Software Inc. for the Atlantis AI product owned by Sumeet kumar jha. Your goal is to provide clear, concise, and accurate responses in a user-friendly format. Begin with a brief summary covering only the essential details, but if the query explicitly requests more detail or if additional context is beneficial, expand your response with thorough explanations, examples, and context. After answering, include your follow-up question on a new line (do not include any literal prefix) to invite further discussion on the topic.';

const freeModals = [
  'deepseek/deepseek-r1-distill-llama-70b:free',
  'deepseek/deepseek-r1:free',
  'deepseek/deepseek-chat:free',
  'meta-llama/llama-3.2-11b-vision-instruct:free',
  'nvidia/llama-3.1-nemotron-70b-instruct:free',
  'undi95/toppy-m-7b:free',
];

// Gemini generation configuration
const GEMINI_CONFIG: GenerationConfig = {
  temperature: 0.9,
  topK: 40,
  topP: 0.8,
  maxOutputTokens: 4096,
  candidateCount: 1,
  stopSequences: ['User:', 'Assistant:'],
};

const SAFETY_SETTINGS: SafetySetting[] = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
];

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
  readonly limiter = new RateLimiterMemory({ points: 100, duration: 60 });

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
    if (!apiKey) throw new Error('GOOGLE_GENAI_API_KEY is not set in environment variables');
    const genAI = new GoogleGenerativeAI(apiKey);
    this.geminiModel = genAI.getGenerativeModel({
      model: 'gemini-1.5-flash',
      generationConfig: GEMINI_CONFIG,
      safetySettings: SAFETY_SETTINGS as unknown as SafetySetting[],
    });
    this.geminiChats = new Map();
    setInterval(() => this.cleanupSessions(), this.SESSION_EXPIRY);
  }

  private cleanupSessions() {
    const now = Date.now();
    for (const [threadId, session] of this.geminiChats.entries()) {
      if (now - session.lastUpdated.getTime() > this.SESSION_EXPIRY) {
        this.geminiChats.delete(threadId);
      }
    }
  }

  public async query(message: string, threadId: string) {
    await this.limiter.consume(threadId);
    const validation = this.validateMessage(message);
    if (!validation.valid) throw new Error(validation.reason);
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

  public async streamDirectCompletion(message: string, ws: WebSocket, threadId?: string): Promise<void> {
    if (!message || typeof message !== 'string') {
      return ws.send(JSON.stringify({ type: 'error', content: 'Invalid message', threadId }));
    }
    let streamedContent = '';
    try {
      const stream = await this.openai.chat.completions.create({
        model: freeModals[freeModals.length - 5],
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: message.trim() },
        ],
        stream: true,
      });
      for await (const chunk of stream) {
        console.log(chunk);
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

  public async geminiCompletion(message: string, ws: WebSocket, threadId: string, userId?: string) {
    if (!message?.trim()) {
      return ws.send(JSON.stringify({ type: 'error', content: 'Invalid message', threadId }));
    }
    const trimmedMessage = message.trim();
    let session: ConversationSession;
    if (this.geminiChats.has(threadId)) {
      session = this.geminiChats.get(threadId)!;
      session.lastUpdated = new Date();
    } else {
      session = await this.initializeNewChat([], userId);
      this.geminiChats.set(threadId, session);
    }

    // Handle "can you look on web" command to use last search query.
    if (trimmedMessage.toLowerCase() === 'can you look on web') {
      if (session.context.lastSearchQuery && session.context.lastSearchQuery.trim().length > 0) {
        await this.handleSearchAndResponse(session.chat, session, session.context.lastSearchQuery, ws, threadId);
        return;
      } else {
        ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: "I don't have any recent query to search for on the web." }));
        return;
      }
    }

    // Handle simple "yes" to answer follow-up.
    if (trimmedMessage.toLowerCase() === 'yes') {
      if (session.context.followUpQuestion && session.context.followUpQuestion.length > 0) {
        const followUpPrompt = session.context.followUpQuestion;
        session.context.followUpQuestion = '';
        const followUpResult = await session.chat.sendMessage(followUpPrompt);
        const answer = followUpResult.response.text();
        ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: answer }));
        return;
      }
    }

    // Analyze if current/real-time information is required.
    const analysisPrompt = `Question/Request: "${trimmedMessage}"

Please analyze if this query requires current or real-time information or fact-checking. If it is creative writing, summarization, or relies on general knowledge, respond with DIRECT.
Only respond with SEARCH if the query explicitly demands up-to-date factual information.
Respond in this format:
DECISION: [SEARCH or DIRECT]
REASON: [One brief sentence explaining why]`;

    const analysisResult = await session.chat.sendMessage(analysisPrompt);
    const analysisResponse = analysisResult.response.text();
    const needsSearch = analysisResponse.includes('DECISION: SEARCH');

    if (needsSearch) {
      await this.handleSearchAndResponse(session.chat, session, trimmedMessage, ws, threadId);
    } else {
      await this.handleNormalChat(session.chat, session.history, trimmedMessage, ws, threadId);
    }
  }

  private async initializeNewChat(history: ChatHistory[], userId?: string): Promise<ConversationSession> {
    const chat = this.geminiModel.startChat({
      history: [],
      generationConfig: GEMINI_CONFIG,
      safetySettings: SAFETY_SETTINGS,
    });
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    try {
      const initResult = await chat.sendMessage(SYSTEM_PROMPT);
      const initResponse = initResult.response.text();
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
          followUpQuestion: '',
          lastSearchQuery: '',
        },
      };
    } catch (error) {
      console.error('Error initializing chat:', error);
      throw new Error('Failed to initialize chat session');
    }
  }

  private async handleSearchAndResponse(
    chat: GeminiChatSession,
    session: ConversationSession,
    message: string,
    ws: WebSocket,
    threadId: string,
  ): Promise<void> {
    // Save current query as lastSearchQuery.
    session.context.lastSearchQuery = message;
    let searchResults: any[];
    const cacheKey = message;
    const cacheEntry = this.searchCache.get(cacheKey);
    if (cacheEntry && Date.now() - cacheEntry.timestamp < 1000 * 60 * 5) {
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

    // If no results found, respond with default message.
    if (!searchResults || searchResults.length === 0) {
      ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: "I'm sorry, I don't have data or access to that information." }));
      return;
    }

    // Build a prompt including search results and context; instruct to add a follow-up question.
    const messageWithContext = `User Query: "${message}"

Search Results: ${JSON.stringify(searchResults, null, 2)}

Previous Context: ${session.context.topic ? `This conversation is about ${session.context.topic}.` : 'No previous topic.'}
Search History: ${session.context.searchHistory.length} searches so far.

Task: Provide a clear and natural response that summarizes the key information. Start with a concise summary, and if additional detail is relevant or explicitly requested, expand your answer with further context, examples, and explanations. On a new line, include your follow-up question (without any prefix) to invite further discussion.`;

    await this.processAndStreamResponse(chat, session.history, messageWithContext, ws, threadId);

    if (!session.context.topic) {
      const topicAnalysis = await chat.sendMessage("Based on our conversation so far, what's the main topic? Respond with just 2-3 words.");
      session.context.topic = topicAnalysis.response.text();
    }
  }

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

Task: Provide a clear and natural response that covers the essential details. Start with a concise summary, and if additional context or examples are relevant, expand your answer with detailed explanations. On a new line, include your follow-up question (without any prefix) to invite further discussion.`;
    await this.processAndStreamResponse(chat, history, messageWithContext, ws, threadId);
    const session = this.geminiChats.get(threadId)!;
    if (!session.context.topic) {
      const topicAnalysis = await chat.sendMessage("Based on our conversation so far, what's the main topic? Respond with just 2-3 words.");
      session.context.topic = topicAnalysis.response.text();
    }
  }

  // Build conversation history from ChatHistory array.
  private buildConversationHistory(history: ChatHistory[]): string {
    return history
      .map(entry => {
        const roleLabel = entry.role === 'user' ? 'User' : 'Assistant';
        return `${roleLabel}: ${entry.parts.map(part => part.text).join(' ')}`;
      })
      .join('\n');
  }

  // Build a brief summary of key topics from history.
  private buildConversationSummary(history: ChatHistory[]): string {
    const topics = new Set<string>();
    history.forEach(entry => {
      if (entry.role === 'model') {
        const text = entry.parts.map(part => part.text.toLowerCase()).join(' ');
        if (text.includes('gold') && text.includes('surat')) {
          topics.add('gold price in Surat, India');
        }
      }
    });
    if (topics.size > 0) {
      return `Reminder: We have been discussing ${Array.from(topics).join(', ')}.`;
    }
    return '';
  }

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
    // Build conversation history and summary.
    const conversationHistory = this.buildConversationHistory(history);
    const conversationSummary = this.buildConversationSummary(history);
    const finalPrompt = `${conversationSummary}\n${conversationHistory}\nAssistant: `;
    try {
      let streamedContent = '';
      const result = await chat.sendMessageStream(finalPrompt);
      for await (const chunk of result.stream) {
        const content = chunk.text();
        if (content) {
          streamedContent += content;
          ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'token', content }));
        }
      }
      if (streamedContent) {
        // Extract any follow-up question.
        const followUp = this.extractFollowUpQuestion(streamedContent);
        session.context.followUpQuestion = followUp || '';
        // Remove unwanted lines (e.g. follow-up markers and specific phrases).
        const unwantedPhrases = ["i still need to know whose education details you're requesting."];
        const displayContent = streamedContent
          .split('\n')
          .filter(line => {
            const lowerLine = line.toLowerCase();
            return !lowerLine.startsWith('follow-up:') && !unwantedPhrases.some(phrase => lowerLine.includes(phrase));
          })
          .join('\n');
        history.push({ role: 'model', parts: [{ text: displayContent }] });
        ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done', content: displayContent }));
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

  private extractFollowUpQuestion(response: string): string | null {
    const lines = response.split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.toLowerCase().startsWith('follow-up:')) {
        return line.substring('follow-up:'.length).trim();
      }
    }
    return null;
  }

  public async streamChatCompletion(message: string, ws: WebSocket, threadId: string) {
    try {
      const completion = await this.openai.chat.completions.create({
        model: 'deepseek/deepseek-r1-distill-llama-70b:free',
        messages: [
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: message },
        ],
        stream: true,
        response_format: { type: 'json_object' },
        reasoning_effort: 'high',
        logprobs: true,
        stream_options: { include_usage: true },
        user: threadId,
        metadata: { thread_id: threadId },
      });
      console.log(completion);
      // Stream handling: process tokens as they arrive
      for await (const chunk of completion) {
        if (chunk.choices && chunk.choices.length > 0) {
          const token = chunk.choices[0]?.delta?.content || ''; // Extract token from response
          if (token) {
            ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'token', content: token }));
          }
        }
      }

      // Send final message indicating completion
      ws.send(JSON.stringify({ role: 'assistant', threadId, type: 'done' }));
    } catch (error) {
      console.error('Error in streaming completion:', error);
      ws.send(JSON.stringify({ role: 'error', threadId, message: 'An error occurred while generating the response.' }));
    }
  }

  public async getSessionInfo(threadId: string): Promise<ConversationSession | null> {
    return this.geminiChats.get(threadId) || null;
  }

  public async clearSession(threadId: string): Promise<void> {
    this.geminiChats.delete(threadId);
  }

  private shouldUseExpensiveModel(message: string): boolean {
    return message.split(/\s+/).length > 50;
  }

  private async withFallback(fn: () => Promise<any>, fallbackFn: () => Promise<any>) {
    try {
      return await fn();
    } catch (error) {
      if (error.statusCode === 429 || error.code === 'rate_limited') {
        return await fallbackFn();
      }
      throw error;
    }
  }

  private validateMessage(message: string): { valid: boolean; reason?: string } {
    const MAX_LENGTH = 1000;
    const BLACKLIST = ['credit card', 'password'];
    if (message.length > MAX_LENGTH) return { valid: false, reason: 'Message too long' };
    if (BLACKLIST.some(term => message.includes(term))) return { valid: false, reason: 'Invalid content' };
    return { valid: true };
  }

  private trackCost(provider: 'google' | 'openrouter', tokens: number) {
    const COST_RATES = {
      google: 0.0000025,
      openrouter: 0.0000018,
    };
    const cost = tokens * COST_RATES[provider];
    // Send cost data to a monitoring system.
  }
}
