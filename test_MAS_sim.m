% Initial test Multi-Agent System
addpath("./Technique/")
addpath("./Agent/")

%% Simple stationary simulation
rng(8988467)
% x = [1 1 1]';
% u = [.1 .3 .7];
u = [1 1 1];
H = [1 1 1];
x_dim = 3;
N = 200;
M = 300;
Na = 6;
% x = 0.5*randn(1,N+2) + 1;
x = ones(1,N+2);
y_sd = 1;
noise = y_sd * randn(1,N,Na,M);
d = zeros(1,N,Na,M);
y = zeros(1,N);


% (28/08/2025) A interpretação aqui está uma salada, x é H, e H é x (que é u), logo a estimativa dá um valor de H (get_H_hat() - classe Agent; e get_H() - classe Technique) e x é tido como conhecido. Tido isto, agora eu preciso corrigir a simulação para refletir nessa minha escolha de valores a serem obtidos/estimados, no intuito de compatibilizar as comparações legadas com as atuais do refactoring. O Refactoring global deve ser deixado para um outro momento, neste momento, eu preciso fazer uma entrega ao Merched.

% y = H*u;
% d = y + noise;

for n = 1:N
    y(n) = u*x(n:n+2)';
    d(1,n,:,:) = y(n) + noise(1,n,:,:);
end

B = [1 1 1 0 0 0;...
     1 1 0 0 0 0;...
     1 0 1 1 0 0;...
     0 0 1 1 1 1;...
     0 0 0 1 1 0;...
     0 0 0 1 0 1;];

y_dim = 1;
% Choose an Agent vector and Run sim
%{ OBS: (Previsão de acordo com a teoria)
%}
switch sim
    case 1
        % 1º Valor alto de RMSE (3 agentes, Wiener, non coop)
        agent_vec = Agent_vector();
    case 2
        % 2º Valor alto de RMSE (Na agentes, Wiener, non coop)
        agent_vec = Agent_vector('n_agents', Na);
    case 3
        % 3º Valor mais baixo de RMSE (Na agents, Wiener, consensus constrain)
        a_vec = repmat(Agent(), [1, Na]);
        for a = 1:Na
            a_vec(a) = Agent();
        end
        agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.consensus_constrain, 'B_matrix', B);
    case 4
        % 4º Valor baixo de RMSE (Na agents, Wiener, single task)
        agent_vec = Agent_vector('n_agents', Na, 'coop_type', Cooperation_type.single_task, 'B_matrix', B);
    case 5
        % 5º Valor mais baixo de RMSE (Na agents, RLS, single task)
        a_vec = repmat(Agent(), [1, Na]);
        for a = 1:Na
            a_vec(a) = Agent('agent_tech', Rls('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'lambda', .85, 'delta', 1e-3));
        end
        agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.single_task, 'B_matrix', B);
    case 6
        % 6º Valor mais baixo de RMSE (Na agents, LMS, single task)
        a_vec = repmat(Agent(), [1, Na]);
        for a = 1:Na
            a_vec(a) = Agent('agent_tech', Lms('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'epsilon', 1e-5, 'mu', .3));
        end
        agent_vec = Agent_vector('agents_vec', a_vec, 'coop_type', Cooperation_type.single_task, 'B_matrix', B);
    case 7
        % 7º Valor alto de RMSE (Na agentes, LMS, non coop)
        a_vec = repmat(Agent(), [1, Na]);
        for a = 1:Na
            a_vec(a) = Agent('agent_tech', Lms('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'epsilon', 1e-5, 'mu', .3));
        end
        agent_vec = Agent_vector('n_agents', Na, 'agents_vec', a_vec);
    
    case 8
        % 8º Valor alto de RMSE (Na agentes, RLS, non coop)
        a_vec = repmat(Agent(), [1, Na]);
        for a = 1:Na
            a_vec(a) = Agent('agent_tech', Rls('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'lambda', .85, 'delta', 1e-3));
        end
        agent_vec = Agent_vector('n_agents', Na, 'agents_vec', a_vec);
    otherwise
        error('Invalid simulation case selected');
end
    
%% Run sim with Monte-Carlo

H_hat_history = zeros(y_dim, x_dim, N, Na, M);
y_hat_history = zeros(y_dim, N, Na, M);
for m = 1:M
    for n = 1:N
        agent_vec.update(d(1,n,:,m), x(n:n+2));
        for a = 1:Na
            H_hat_history(:,:,n,a,m) = agent_vec.agents_vec(a).get_H_hat();
            y_hat_history(:,n,a,m) = agent_vec.agents_vec(a).get_y_hat();
        end
    end
    agent_vec.reset();
    if (mod(m/M, .05) == 0)
        fprintf('Completed %.0f%% of Monte-Carlo realizations.\n', m/M*100);
    end
end

%% Plots

figure(1)
len = N;
n = 1:len;
plot(n, d(1,1:len,1,1), 'b')
hold on
plot(n, y_hat_history(1,1:len,1,1), 'r--', 'LineWidth', 1.5)
plot(n, y_hat_history(1,1:len,2,1), 'g-.', 'LineWidth', 1.5)
plot(n, y_hat_history(1,1:len,3,1), 'y--', 'LineWidth', 1.5)
xlabel('Time')
ylabel('y_{hat}')
title('A realization')
legend('d', 'a1', 'a2', 'a3')
hold off
grid on

n = 1:N;
% plot(n, reshape(y_hat_history(:,1,:), [1, N]), 'r--');
% plot(n, reshape(y_hat_history(:,2,:), [1, N]), 'g--');
% plot(n, reshape(y_hat_history(:,3,:), [1, N]), 'k--');

%% The error
a = 4;
e1 =  (y_hat_history(1,:,a,:) - d(1,:,a,:)).^2;
e11 =  (y_hat_history(1,:,a,1) - d(1,:,a,1)).^2;
e1m = mean(e1,4);
% e1 = abs(reshape(y_hat_history(:,1,:), [1, N]) - u*x);
e2 = zeros(1,N,M);
for m = 1:M
    for i = 1:N
        e2(1,i,m) = norm(H_hat_history(:,:,i,a,m) - u);
    end
end
e21 = e2(1,:,1);
e2m = mean(e2,3);
    
figure(2)
plot(n,10*log10(e1m),'b')
% hold on
% plot(n,10*log10(e11),'r')
% hold off
tit = sprintf('Test: MSE. Agente = %d', a);
title(tit)
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

figure(3)
plot(n,20*log10(e2m),'b')
% hold on
% plot(n,10*log10(e21),'r')
% hold off
tit = sprintf('Test: MSD. Agente = %d', a);
title(tit)
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

%% Benchmark

y_ticks = -40:5:0;
figure(4)
plot(n,10*log10(e1m_wiener_noncoop_a4),'r')
hold on
plot(n,10*log10(e1m_wiener_a4),'b')
hold off
ylabel('MSE')
xlabel('n')
set(gca, 'YLim', [-40 0], 'YTick', y_ticks)
legend('noncoop', 'ATC w_{hat}')
grid on

figure(5)
plot(n,20*log10(e2m_wiener_noncoop_a4),'r')
hold on
plot(n,20*log10(e2m_wiener_a4),'b')
hold off
ylabel('MSD')
xlabel('n')
set(gca, 'YLim', [-40 0], 'YTick', y_ticks)
legend('noncoop', 'ATC w_{hat}')
grid on