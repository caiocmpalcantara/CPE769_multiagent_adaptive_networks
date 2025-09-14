% Test script for Agent with KF (Kalman Filter)
% Based on test_agent.m structure and test_kalman.m case 1 simulation

global DEBUG_MODE;
DEBUG_MODE = false;

addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")
addpath("./Utils/")

%% Simulation Setup (similar to test_kalman.m case 1)
sim = 1; % Use case 1 from test_kalman.m

switch sim
    case 1
        % Simple stationary simulation
        x_dim = 3;
        y_dim = 1;
        y_sd = 0.2;

        u = [.1 .3 .7]';   % The true "state" we want to estimate
        N = 200;
        n = 1:N;
        x = 0.5*randn(1,N+2) + 1;  % Input signal
        noise = y_sd * randn(1,N);
        d = zeros(1,N);
        y = zeros(1,N);
        H = zeros(N, x_dim);

        % Generate observations
        for i = 1:N
            H(i,:) = x(i:i+2);
            y(i) = H(i,:) * u;
            d(i) = y(i) + noise(i);
        end

    otherwise
        error('Invalid simulation case selected');
end

%% Figures
figure(1)
clf;
plot(n, d, 'b')
xlabel('Time')
ylabel('Observation')
title('Testing Agent with KF')
hold on
grid on

%% Create Agent with KF
% System model setup
Q = zeros(x_dim);
A = eye(x_dim);
model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

% Observation model setup
H_matrix = H(1,:);  % Initial H matrix
R = y_sd^2;         % Measurement noise variance

% Create KF technique
kf_technique = KF('x_dim', x_dim, 'y_dim', y_dim, ...
                      'H_matrix', H_matrix, 'R_matrix', R, ...
                      'Pa_init', {'delta', 0.1}, ...
                      'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                      'system_model', model_sys);

% Create Agent with KF technique
agent = Agent('x_dim', x_dim, 'y_dim', y_dim, 'agent_tech', kf_technique);

%% Run simulation
H_hat_history = zeros(y_dim, x_dim, N);
y_hat_history = zeros(y_dim, N);
xp_hat_history = zeros(x_dim, N);  % Posterior state estimates
xa_hat_history = zeros(x_dim, N);  % Prior state estimates
P_trace_history = zeros(1, N);     % Trace of posterior covariance

fprintf('Running Agent simulation...\n');

for i = 1:N
    % Store current estimates before update
    H_hat_history(:,:,i) = agent.agent_technique.get_H();
    
    % Perform self-learning step with current measurement
    agent.self_learning_step('measurement', d(i));
    
    % Store results after update
    y_hat_history(:,i) = agent.get_y_hat();
    xp_hat_history(:,i) = agent.get_posterior_state();
    xa_hat_history(:,i) = agent.get_prior_state();
    
    % Store covariance trace if available
    P = agent.get_posterior_covariance();
    if ~isempty(P)
        P_trace_history(i) = trace(P);
    end
    
    % Progress indicator
    if mod(i, 50) == 0
        fprintf('Processed %d/%d samples\n', i, N);
    end
end

% Plot prediction
plot(n, y_hat_history, 'r', 'LineWidth', 1.5);
legend('Observations', 'Kalman Predictions', 'Location', 'best');
hold off

%% Performance Analysis

% Prediction error
e_pred = abs(y_hat_history - y');
e_pred_mean = mean(e_pred);

% Parameter estimation error (H matrix estimation)
e_param = zeros(1, N);
for i = 1:N
    e_param(i) = norm(H_hat_history(:,:,i) - u');
end

% State estimation error
e_state = zeros(1, N);
for i = 1:N
    e_state(i) = norm(xp_hat_history(:,i) - u);
end

%% Error Plots
figure(2)
subplot(3,1,1)
plot(n, e_pred, 'b', 'LineWidth', 1.5)
title('Prediction Error |y\_hat - y|')
ylabel('Error')
xlabel('Time')
grid on

subplot(3,1,2)
plot(n, e_param, 'r', 'LineWidth', 1.5)
title('Parameter Estimation Error ||H\_hat - u||')
ylabel('Error')
xlabel('Time')
grid on

subplot(3,1,3)
plot(n, e_state, 'g', 'LineWidth', 1.5)
title('State Estimation Error ||x\_hat - u||')
ylabel('Error')
xlabel('Time')
grid on

%% Covariance Analysis
if any(P_trace_history > 0)
    figure(3)
    plot(n, P_trace_history, 'm', 'LineWidth', 1.5)
    title('Posterior Covariance Trace Evolution')
    ylabel('Trace(P)')
    xlabel('Time')
    grid on
end

%% Performance Metrics
fprintf('\n=== Agent Performance Metrics ===\n');
fprintf('Mean Prediction Error: %.6f\n', e_pred_mean);
fprintf('Final Parameter Error: %.6f\n', e_param(end));
fprintf('Final State Error: %.6f\n', e_state(end));

if any(P_trace_history > 0)
    fprintf('Final Covariance Trace: %.6f\n', P_trace_history(end));
end

% Convergence analysis
convergence_threshold = 0.1;
convergence_idx = find(e_param < convergence_threshold, 1);
if ~isempty(convergence_idx)
    fprintf('Parameter convergence achieved at sample: %d\n', convergence_idx);
else
    fprintf('Parameter convergence not achieved within threshold %.3f\n', convergence_threshold);
end

%% State Evolution Plot
figure(4)
plot(n, xp_hat_history(1,:), 'r-', 'LineWidth', 1.5)
hold on
plot(n, xp_hat_history(2,:), 'g-', 'LineWidth', 1.5)
plot(n, xp_hat_history(3,:), 'b-', 'LineWidth', 1.5)
plot(n, u(1)*ones(size(n)), 'r--', 'LineWidth', 1)
plot(n, u(2)*ones(size(n)), 'g--', 'LineWidth', 1)
plot(n, u(3)*ones(size(n)), 'b--', 'LineWidth', 1)
title('State Estimation Evolution')
xlabel('Time')
ylabel('State Values')
legend('x_1 estimate', 'x_2 estimate', 'x_3 estimate', ...
       'x_1 true', 'x_2 true', 'x_3 true', 'Location', 'best')
grid on
hold off

fprintf('\nAgent test completed successfully!\n');
