import WebSocket from 'ws';

interface AIMessage {
  type: 'token' | 'tool' | 'done' | 'error' | 'connected';
  content: string;
  role?: string;
  threadId?: string;
}

interface GeminiRequest {
  type: 'gemini';
  content: string;
  threadId?: string;
}

const ws = new WebSocket('ws://localhost:3000/ai');

const sendGeminiMessage = (content: string, threadId?: string): void => {
  const message: GeminiRequest = {
    type: 'gemini',
    content,
    threadId,
  };
  ws.send(JSON.stringify(message));
};

ws.onopen = (): void => {
  console.log('Connected to AI service');
  sendGeminiMessage('What is the latest news about AI technology?', 'session_123');
};

ws.onmessage = (event: WebSocket.MessageEvent): void => {
  const message: AIMessage = JSON.parse(event.data.toString());

  switch (message.type) {
    case 'token':
      // Handle streaming tokens
      process.stdout.write(message.content);
      break;

    case 'tool':
      // Handle search notifications
      console.log('\nSearch Info:', message.content);
      break;

    case 'done':
      console.log('\nFinal Response:', message.content);
      // Ask follow-up question
      sendGeminiMessage('Can you explain more about that?', 'session_123');
      break;

    case 'error':
      console.error('Error:', message.content);
      break;
  }
};

ws.onerror = (error: WebSocket.ErrorEvent): void => {
  console.error('WebSocket error:', error);
};

ws.onclose = (): void => {
  console.log('Disconnected from AI service');
};
