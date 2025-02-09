import { App } from '@/app';
import { AuthRoute } from '@routes/auth.route';
import { UserRoute } from '@routes/users.route';
import { AgentRoute } from '@routes/agent.route';
import { ValidateEnv } from '@utils/validateEnv';
import { AgentWebSocket } from './websocket/agent.websocket';

ValidateEnv();

const app = new App([new UserRoute(), new AuthRoute(), new AgentRoute()]);
const server = app.listen();

// Initialize WebSocket server
const wsServer = new AgentWebSocket(server);

// Export for use in other parts of your application if needed
export { wsServer, app };
