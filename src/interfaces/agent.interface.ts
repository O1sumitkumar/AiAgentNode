import { ChatSession as GeminiChatSession } from '@google/generative-ai';

export interface ChatHistory {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

export interface ChatSession {
  chat: GeminiChatSession;
  history: ChatHistory[];
}

export interface ConversationSession extends ChatSession {
  sessionId: string;
  userId?: string;
  createdAt: Date;
  lastUpdated: Date;
  context: {
    followUpQuestion?: string;
    followUpAnswer?: string;
    topic?: string;
    searchHistory: Array<{
      query: string;
      results: any[];
      timestamp: Date;
    }>;
    messageCount: number;
    lastInteraction: Date;
    summary?: string;
    lastSearchQuery?: string;
  };
}
