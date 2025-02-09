import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { App } from '@/app';
import { AuthRoute } from '@routes/auth.route';
import { UserRoute } from '@routes/users.route';
import { AgentRoute } from '@routes/agent.route';
import { ValidateEnv } from '@utils/validateEnv';
import { setupAgentWebSocket } from './websocket/agent.websocket';

ValidateEnv();

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ai' });

wss.on('connection', ws => {
  setupAgentWebSocket(ws);
});

const appInstance = new App([new UserRoute(), new AuthRoute(), new AgentRoute()]);

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// Export for use in other parts of your application if needed
export { appInstance };
