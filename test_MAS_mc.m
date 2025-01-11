% Initial test Multi-Agent System
addpath("./Technique/")
addpath("./Agent/")

%% Simple stationary simulation
rng(8988466)
x = [1 1 1]';
x_dim = 3;
u = [1 1 1];
N = 1000;
M = 200;
Na = 6;
noise = randn(1,N,Na,M);

d = u*x + noise;
y_dim = 1;

% Choose an Agent vector and Run sim
%{ OBS: (Previsão de acordo com a teoria)
%}
%% 1º Valor alto de RMSE (3 agentes, Wiener, non coop)
agent_vec = Agent_vector();
%% 2º Valor alto de RMSE (Na agentes, Wiener, non coop)
agent_vec = Agent_vector('n_agents', Na);
%% 3º Valor mais baixo de RMSE (Na agents, Wiener, consensus constrain)
a_vec = repmat(Agent(), [1, Na]);
for a = 1:Na
    a_vec(a) = Agent();
end
agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.consensus_constrain, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                              1 1 0 0 0 0;...
                                                                                                              1 0 1 1 0 0;...
                                                                                                              0 0 1 1 1 1;...
                                                                                                              0 0 0 1 1 0;...
                                                                                                              0 0 0 1 0 1;]);
%% 4º Valor baixo de RMSE (Na agents, Wiener, single task)
agent_vec = Agent_vector('n_agents', Na, 'coop_type', Cooperation_type.single_task, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                1 1 0 0 0 0;...
                                                                                                1 0 1 1 0 0;...
                                                                                                0 0 1 1 1 1;...
                                                                                                0 0 0 1 1 0;...
                                                                                                0 0 0 1 0 1;]);
%% 5º Valor mais baixo de RMSE (Na agents, RLS, single task)
a_vec = repmat(Agent(), [1, Na]);
for a = 1:Na
    a_vec(a) = Agent('agent_tech', Rls('lambda', 0.88));
end
agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.single_task, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                      1 1 0 0 0 0;...
                                                                                                      1 0 1 1 0 0;...
                                                                                                      0 0 1 1 1 1;...
                                                                                                      0 0 0 1 1 0;...
                                                                                                      0 0 0 1 0 1;]);
%% 6º Valor mais baixo de RMSE (Na agents, LMS, single task)
a_vec = repmat(Agent(), [1, Na]);
for a = 1:Na
    a_vec(a) = Agent('agent_tech', Lms('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'epsilon', 1e-5, 'mu', .3));
end
agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.single_task, 'B_matrix', [1 1 1 0 0 0;...
                                                                                                      1 1 0 0 0 0;...
                                                                                                      1 0 1 1 0 0;...
                                                                                                      0 0 1 1 1 1;...
                                                                                                      0 0 0 1 1 0;...
                                                                                                      0 0 0 1 0 1;]);
%% Run sim with Monte-Carlo

H_hat_history = zeros(y_dim, x_dim, N, Na, M);
y_hat_history = zeros(y_dim, N, Na, M);
for m = 1:M
    for n = 1:N
        agent_vec.update(d(1,n,:,m),x);
        for a = 1:Na
            H_hat_history(:,:,n,a,m) = agent_vec.agents_vec(a).get_H_hat();
            y_hat_history(:,n,a,m) = agent_vec.agents_vec(a).get_y_hat();
        end
    end    
end

%% Plots

figure(1)
n = 1:N;
plot(n, d(1,:,1,1), 'b')
hold on
plot(n, y_hat_history(1,:,1,1), 'r--', 'LineWidth', 1.5)
plot(n, y_hat_history(1,:,2,1), 'g-.', 'LineWidth', 1.5)
plot(n, y_hat_history(1,:,3,1), 'y--', 'LineWidth', 1.5)
xlabel('Time')
ylabel('Observation')
title('A realization')
legend('d', 'a1', 'a2', 'a3')
hold off
grid on

n = 1:N;
% plot(n, reshape(y_hat_history(:,1,:), [1, N]), 'r--');
% plot(n, reshape(y_hat_history(:,2,:), [1, N]), 'g--');
% plot(n, reshape(y_hat_history(:,3,:), [1, N]), 'k--');

%% The error
a = 1;
e1 =  (y_hat_history(1,:,a,:) - u*x).^2;
e1m = mean(e1,4);
% e1 = abs(reshape(y_hat_history(:,1,:), [1, N]) - u*x);
e2 = zeros(1,N,M);
for m = 1:M
    for i = 1:N
        e2(1,i,m) = norm(H_hat_history(:,:,i,a,m) - u);
    end
end
e2m = mean(e2,3);
    
figure(2)
plot(n,10*log10(e1m),'b')
title('Test: Error.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

figure(3)
plot(n,10*log10(e2m),'r')
title('Test: Error.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on
