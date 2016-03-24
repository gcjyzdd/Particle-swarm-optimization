% Particle swarm optimization demo
%
% By Changjie Guan
% at TU/e, the Netherlands

CostFunction = @(x)x(1)*exp(-norm(x)^2);
lb=[-10 -10];
ub=[10 10];
SwarmSize=100;
MaxIt=200;
tol=1e-5;
[best_p,best_f]=ParticleSwarmOpt(CostFunction,SwarmSize,lb,ub,MaxIt,tol)