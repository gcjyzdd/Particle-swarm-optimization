function [best_p,best_f]=ParticleSwarmOpt(CostFunction,SwarmSize,lb,ub,MaxIt,tol)
%% [best_p,best_f]=ParticleSwarmOpt(f,lb,ub,tol)
% Particle swarm optimization
% input:
%   f-  cost function, calculate fitness of each particle, a function handle     
%   SwarmSize-  number of particles
%   lb- lower bound of the decision variable, a row vector
%   ub- upper bound of the decision variable, a row vector
%   MaxIt-  the maximum number of iterations, recommand 200*#decision
%   variable
%   tol-tolerance of the global optimum
%
% output:
%   best_p- the best decision variable
%   best_f- the best fitness
%
%
%
% By Changjie Guan
% at TU/e, the Netherlands




nVar=length(lb);            % Number of Decision Variables


% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.5;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient


% Velocity Limits
VelMax=0.1*(ub-lb);
VelMin=-VelMax;

VELMAX=bsxfun(@times,ones(SwarmSize,1),VelMax);
VELMIN=bsxfun(@times,ones(SwarmSize,1),VelMin);
VARMAX=bsxfun(@times,ones(SwarmSize,1),ub);
VARMIN=bsxfun(@times,ones(SwarmSize,1),lb);

% initialization
Velocity=zeros(SwarmSize,nVar);
Cost    =zeros(SwarmSize,1);
Position=VARMIN+bsxfun(@times,rand(SwarmSize,nVar),(ub-lb));

% use parfor below if CostFunction takes a long time
for i=1:SwarmSize
    Cost(i)=CostFunction(Position(i,:));
end
FCalls=SwarmSize;
PbestP=Position;
PbestC=Cost;
[temp,ind]=min(Cost);
GbestC=temp(1);
GbestP=Position(ind(1),:);


fprintf('Iteration    Best        Mean         FunCalls\n');
% PSO Main Loop

for it=1:MaxIt
    
    % update velocity
    Velocity=w*Velocity+c1*rand(SwarmSize,nVar).*(PbestP-Position)...
        +c2*rand(SwarmSize,nVar).*(bsxfun(@times,ones(SwarmSize,1),GbestP)-Position);
    
    % apply velocity limits    
    Velocity=bsxfun(@min,Velocity,VELMAX);
    Velocity=bsxfun(@max,Velocity,VELMIN);
    
    % update position
    Position=Position+Velocity;
    
    % apply velocity mirror effect
    IND=(Position>VARMAX)|(Position<VARMIN);
    Velocity(IND)=-Velocity(IND);
    
    % apply position limits
    Position=bsxfun(@min,Position,VARMAX);
    Position=bsxfun(@max,Position,VARMIN);
    
    % update cost
    % use parfor below if CostFunction takes a long time
    for i=1:SwarmSize
        Cost(i)=CostFunction(Position(i,:));
    end
    
    FCalls=FCalls+SwarmSize;
    
    % update personal best
    IND2=PbestC>Cost;
    PbestC(IND2)=Cost(IND2);
    PbestP(IND2,:)=Position(IND2,:);
    % update global best
    [temp,ind]=min(Cost);
    del=abs(GbestC-temp(1));
    GbestC=temp(1);
    GbestP=Position(ind(1),:);
        
    BestCost(it)=GbestC;
    
    if mod(it,60)==0
        fprintf('\nIteration    Best        Mean         FunCalls\n');
    end
    fprintf('%i          %3.3f      %3.3f       %i\n',it,GbestC,mean(PbestC),FCalls);
    
    w=w*wdamp;
    
    if (del/GbestC)<1e-6 && abs(mean(PbestC)-GbestC)<tol
        break;
    end
end

best_p = GbestP;
best_f = GbestC;

