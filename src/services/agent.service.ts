import { Service } from 'typedi';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import OpenAI from 'openai';
import { MemorySaver } from '@langchain/langgraph';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { WebSocket } from 'ws';

const SYSTEM_PROMPT = `You are Claude, a friendly and helpful AI assistant created by Crisp for their Crisp AI product. While you're knowledgeable, you maintain a humble approach and aren't afraid to acknowledge when you're uncertain.

You have a good sense of humor and can engage in light-hearted banter while remaining professional and focused on helping users effectively.

You're excellent at:
- Understanding and maintaining context throughout conversations
- Breaking down complex topics into simpler explanations
- Admitting when you need to verify information

You have access to these powerful tools:
- Web Search: You can search the internet for current information via the Tavily search engine
- [Additional tools will be automatically available based on configuration]

Important guidelines:
- Always verify critical information using your tools when needed
- Maintain context from previous messages in the conversation
- Be transparent about your capabilities and limitations
- Feel free to add appropriate humor when it helps make explanations more engaging
- Stay focused on providing accurate, helpful responses while being personable

Remember: You're part of the Crisp AI Team, and your goal is to provide the best possible assistance while being authentic and approachable.
`;

@Service()
export class AgentService {
  private readonly agent;
  private readonly getWebSearchResults: TavilySearchResults;
  private readonly agentCheckpointer: MemorySaver;
  private readonly openai: OpenAI;
  private readonly systemPrompt: string;

  constructor() {
    this.getWebSearchResults = new TavilySearchResults({ maxResults: 3, apiKey: process.env.TAVILY_API_KEY, includeImages: true });

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
  }

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

  public async streamQuery(message: string, threadId: string | undefined, ws: WebSocket): Promise<void> {
    try {
      let streamedContent = '';

      const response = await this.agent.invoke(
        {
          messages: [
            { role: 'system', content: this.systemPrompt },
            { role: 'user', content: message },
          ],
        },
        {
          configurable: { thread_id: threadId },
          callbacks: [
            {
              handleLLMNewToken(token: string) {
                streamedContent += token;
                ws.send(
                  JSON.stringify({
                    type: 'token',
                    content: token,
                    threadId: threadId,
                  }),
                );
              },
              handleToolEnd(output: string) {
                ws.send(
                  JSON.stringify({
                    type: 'tool',
                    content: output,
                    threadId: threadId,
                  }),
                );
              },
            },
          ],
        },
      );

      // Send completion message
      ws.send(
        JSON.stringify({
          type: 'done',
          content: streamedContent,
          threadId: threadId,
        }),
      );
    } catch (error) {
      console.error('Error in streamQuery:', error);
      ws.send(
        JSON.stringify({
          type: 'error',
          content: error instanceof Error ? error.message : 'Error processing query',
          threadId: threadId,
        }),
      );
    }
  }

  private async startStreaming(message: string, threadId?: string) {
    // Implement your streaming logic here
    // This could involve calling an AI service, database queries, etc.
    return 'Your response here';
  }

  // Optional: Direct access to OpenAI client for other operations
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
}
