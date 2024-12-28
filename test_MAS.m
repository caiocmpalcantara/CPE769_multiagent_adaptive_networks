% Initial test Multi-Agent System
addpath("./Technique/")
addpath("./Agent/")

%% Simple stationary simulation
x = [1 1 1]';
x_dim = 3;
u = [1 1 1];
N = 200;
n = 1:N;
Na = 6;
noise = randn(Na,N);

d = u*x + noise;
y_dim = 1;

%% Figures
figure(1)
plot(n,d(1,:),'b')
xlabel('Time')
ylabel('Observation')
title('Testing LMS Class')
hold on
grid on

% Choose an Agent vector and Run sim
%{ OBS: (Previsão de acordo com a teoria)
%}
%% 1º Valor alto de RMSE (3 agentes, non coop)
 agent_vec = Agent_vector();
%% 2º Valor alto de RMSE (6 agentes, non coop)
agent_vec = Agent_vector('n_agents', 6);
%% 3º Valor mais baixo de RMSE (6 agents, consensus constrain)
a_vec = repmat(Agent(), [1, 6]);
for a = 1:6
    a_vec(a) = Agent();
end
agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.consensus_constrain, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                              1 1 0 0 0 0;...
                                                                                                              1 0 1 1 0 0;...
                                                                                                              0 0 1 1 1 1;...
                                                                                                              0 0 0 1 1 0;...
                                                                                                              0 0 0 1 0 1;]);
%% 4º Valor baixo de RMSE (6 agents, single task)
agent_vec = Agent_vector('n_agents', 6, 'coop_type', Cooperation_type.single_task, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                1 1 0 0 0 0;...
                                                                                                1 0 1 1 0 0;...
                                                                                                0 0 1 1 1 1;...
                                                                                                0 0 0 1 1 0;...
                                                                                                0 0 0 1 0 1;]);
%% 5º Valor mais baixo de RMSE (6 agents, consensus constrain, RLS)
a_vec = repmat(Agent(), [1, 6]);
for a = 1:6
    a_vec(a) = Agent('agent_tech', Rls('lambda', 0.88));
end
agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.single_task, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                              1 1 0 0 0 0;...
                                                                                                              1 0 1 1 0 0;...
                                                                                                              0 0 1 1 1 1;...
                                                                                                              0 0 0 1 1 0;...
                                                                                                              0 0 0 1 0 1;]);
%% Run sim
Na = agent_vec.n_agents;
H_hat_history = zeros(y_dim, x_dim, Na, N);
y_hat_history = zeros(y_dim, Na, N);
obs = d';
for n = 1:N
    agent_vec.update(obs(n,:),x);
    for a = 1:Na
        H_hat_history(:,:,a,n) = agent_vec.agents_vec(a).get_H_hat();
        y_hat_history(:,a,n) = agent_vec.agents_vec(a).get_y_hat();
    end
end

%% Plots
n = 1:N;
plot(n, reshape(y_hat_history(:,1,:), [1, N]), 'r--');
plot(n, reshape(y_hat_history(:,2,:), [1, N]), 'g--');
plot(n, reshape(y_hat_history(:,3,:), [1, N]), 'k--');
legend('obs', 'a1', 'a2', 'a3')
hold off

%% The error
e1 = abs(reshape(y_hat_history(:,1,:), [1, N]) - u*x);
e2 = zeros(1,N);
for i = 1:N
    e2(i) = norm(H_hat_history(:,:,1,i) - u);
end
figure(2)
plot(n,e1,'b')
hold on
plot(n,e2,'r')
title('Test: Error.')
ylabel('e[n]')
xlabel('n')
legend('y', 'H')
grid on
hold off
