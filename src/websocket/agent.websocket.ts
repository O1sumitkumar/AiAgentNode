import { WebSocket } from 'ws';
import { Container } from 'typedi';
import { AgentService } from '@services/agent.service';

interface Message {
  type: 'agent' | 'direct' | 'gemini';
  content: string;
  threadId?: string;
}

export function setupAgentWebSocket(ws: WebSocket): void {
  const agentService = Container.get(AgentService);
  console.log('New WebSocket connection established');

  ws.on('message', async (data: string) => {
    try {
      const message: Message = JSON.parse(data);

      // Validate message structure
      if (!message.type || !message.content) {
        throw new Error('Invalid message format: type and content are required');
      }

      // Validate content is non-empty string
      if (typeof message.content !== 'string' || !message.content.trim()) {
        throw new Error('Message content must be a non-empty string');
      }

      switch (message.type) {
        case 'agent':
          await agentService.streamQuery(message.content.trim(), message.threadId, ws);
          break;

        case 'direct':
          await agentService.streamDirectCompletion(message.content.trim(), ws, message.threadId);
          break;

        case 'gemini':
          await agentService.geminiCompletion(message.content.trim(), ws, message.threadId || '');
          break;

        default:
          ws.send(
            JSON.stringify({
              type: 'error',
              role: 'assistant',
              content: 'Invalid message type. Must be either "agent", "direct", or "gemini".',
            }),
          );
      }
    } catch (error) {
      console.error('WebSocket error:', error);
      ws.send(
        JSON.stringify({
          type: 'error',
          role: 'assistant',
          content: error instanceof Error ? error.message : 'Error processing message',
        }),
      );
    }
  });

  ws.on('error', error => {
    console.error('WebSocket error:', error);
    ws.send(
      JSON.stringify({
        type: 'error',
        content: 'WebSocket error occurred',
      }),
    );
  });

  // Send initial connection success message
  ws.send(
    JSON.stringify({
      type: 'connected',
      content: 'Connected to AI service',
    }),
  );
}
