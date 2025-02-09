import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';
import { Container } from 'typedi';
import { AgentService } from '@services/agent.service';

interface WebSocketMessage {
  type: 'query' | 'stop';
  message?: string;
  threadId?: string;
}

interface WebSocketResponse {
  type: 'token' | 'tool' | 'done' | 'error' | 'connection';
  content?: string;
  status?: string;
  threadId?: string;
}

export class AgentWebSocket {
  private readonly wss: WebSocketServer;
  private readonly agentService: AgentService;
  private readonly clients: Map<WebSocket, { threadId?: string }>;

  constructor(server: Server) {
    this.wss = new WebSocketServer({ server });
    this.agentService = Container.get(AgentService);
    this.clients = new Map();
    this.initialize();
  }

  private initialize(): void {
    this.wss.on('connection', (ws: WebSocket) => {
      console.log('New WebSocket connection established');
      this.clients.set(ws, {});

      ws.on('message', async (message: string) => {
        try {
          const data = this.parseMessage(message);
          await this.handleMessage(ws, data);
        } catch (error) {
          console.error('WebSocket error:', error);
          this.sendError(ws, error instanceof Error ? error.message : 'Unknown error occurred');
        }
      });

      ws.on('close', () => {
        console.log('Client disconnected from agent websocket');
        this.clients.delete(ws);
      });

      ws.on('error', error => {
        console.error('WebSocket error:', error);
        this.sendError(ws, 'WebSocket error occurred');
      });

      // Send initial connection success message
      ws.send(JSON.stringify({ type: 'connection', status: 'connected' }));
    });
  }

  private parseMessage(message: string): WebSocketMessage {
    try {
      const data = JSON.parse(message);
      if (!data.type) {
        throw new Error('Message type is required');
      }
      return data as WebSocketMessage;
    } catch (error) {
      console.error('Invalid message format:', error);
      throw new Error('Invalid message format');
    }
  }

  private async handleMessage(ws: WebSocket, data: WebSocketMessage): Promise<void> {
    switch (data.type) {
      case 'query':
        if (!data.message) {
          throw new Error('Message content is required for queries');
        }
        const clientInfo = this.clients.get(ws);
        if (clientInfo) {
          clientInfo.threadId = data.threadId;
        }
        await this.agentService.streamQuery(data.message, data.threadId, ws);
        break;

      case 'stop':
        // Implement stop functionality if needed
        // this.agentService.stopStream(data.threadId);
        break;

      default:
        throw new Error('Unsupported message type');
    }
  }

  private sendError(ws: WebSocket, message: string): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'error',
          content: message,
        }),
      );
    }
  }

  // Public method to broadcast messages to all connected clients
  public broadcast(message: any): void {
    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  }
}
