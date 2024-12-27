addpath("./Technique/")
addpath("./Agent/")

x = [1 1 1]';
u = [1 1 1];
N = 200;
n = 1:N;
noise = randn(1,N);

d = u*x + noise;

%% Figures
figure(1)
plot(n,d,'b')
xlabel('Time')
ylabel('Observation')
title('Testing LMS Class')
hold on
grid on

%% Criando Agente
agent = [Agent() Agent('agent_tech', Lms('H_ini', [0 0 0]))];
H_hat_history = zeros(agent(2).y_dim, agent(2).x_dim, N);
y_hat_history = zeros(agent(2).y_dim, N);

for i = 1:N
    H_hat_history(:,:,i) = agent(2).get_H_hat();
    agent(2) = agent(2).self_learning_step(x, d(i));
    y_hat_history(:,i) = agent(2).get_y_hat();
    % disp(agent.state_buffer)
    % disp(agent.obs_buffer)
end

plot(n, y_hat_history, 'r');
hold off

%% The error
e1 = abs(y_hat_history - u*x);
e2 = zeros(1,N);
for i = 1:N
    e2(i) = norm(H_hat_history(:,:,i) - u);
end
figure(2)
plot(n,e1,'b')
hold on
plot(n,e2,'r')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on
hold off