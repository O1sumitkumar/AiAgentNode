import { AgentService } from '@/services/agent.service';
import { NextFunction, Request, Response } from 'express';
import Container from 'typedi';

export class AgentController {
  public agent = Container.get(AgentService);

  public getAgent = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const agent = await this.agent.query(req.body.message, req.body.threadId);
      res.status(200).json({ data: agent, message: 'Response from AI Interaction Agent' });
    } catch (error) {
      next(error);
    }
  };

  public directCompletion = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const agent = await this.agent.directCompletion(req.body.message);
      res.status(200).json({ data: agent, message: 'Response from direct completion' });
    } catch (error) {
      next(error);
    }
  };
}
