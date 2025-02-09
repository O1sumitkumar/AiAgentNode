import { Router } from 'express';
import { AgentController } from '@/controllers/agent.controller';
import { Routes } from '@/interfaces/routes.interface';

export class AgentRoute implements Routes {
  public path = '/agent';
  public router: Router;
  public agent = new AgentController();

  constructor() {
    this.router = Router();
    this.initializeRoutes();
  }

  private initializeRoutes() {
    this.router.post(`${this.path}`, this.agent.getAgent);
    this.router.post(`${this.path}/direct`, this.agent.directCompletion);
  }
}
