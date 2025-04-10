import { App } from '@/app';
import { AgentRoute } from '@routes/agent.route';
import { AuthRoute } from '@routes/auth.route';
import { UserRoute } from '@routes/users.route';
import { ValidateEnv } from '@utils/validateEnv';
// import { createServer } from 'http';
// import { WebSocketServer } from 'ws';
// import { setupAgentWebSocket } from './websocket/agent.websocket';

ValidateEnv();

const appInstance = new App([new UserRoute(), new AuthRoute(), new AgentRoute()]);

appInstance.listen();

// const server = createServer(appInstance.app);
// const wss = new WebSocketServer({ server, path: '/ai' });

// wss.on('connection', (ws: any) => {
//   setupAgentWebSocket(ws);
// });

// const PORT = process.env.PORT || 3000;
// server
//   .listen(PORT, () => {
//     console.log(`Server is running on port ${PORT}`);
//   })
//   .on('error', error => {
//     console.error('Error starting the server:', error);
//     process.exit(1); // Exit the process if the server fails to start
//   });

// Export for use in other parts of your application if needed
export { appInstance };
