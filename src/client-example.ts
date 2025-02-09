import { WebSocket } from 'ws';

const ws: WebSocket = new WebSocket('ws://your-server-url');

ws.onopen = (): void => {
  console.log('Connected to WebSocket');

  // Send a query
  ws.send(
    JSON.stringify({
      type: 'query',
      message: 'Hello, how are you?',
      threadId: 'some-thread-id',
    }),
  );
};

ws.onmessage = (event: MessageEvent): void => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'token':
      // Handle streaming token
      console.log('Received token:', data.content);
      break;
    case 'tool':
      // Handle tool output
      console.log('Tool output:', data.content);
      break;
    case 'done':
      // Handle completion
      console.log('Final response:', data.content);
      break;
    case 'error':
      // Handle error
      console.error('Error:', data.content);
      break;
  }
};

ws.onerror = (error: Event): void => {
  console.error('WebSocket error:', error);
};

ws.onclose = (): void => {
  console.log('Disconnected from WebSocket');
};
