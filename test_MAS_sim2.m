% Multi-Agent System test with Agent2 and KF_diff (Kalman Filter)
% Based on test_MAS_sim.m structure with Agent2 and Kalman filtering

addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")

%% Simulation Setup (similar to test_MAS_sim.m)
rng(8988467)  % For reproducible results

% System parameters
u = [.1 .3 .7];     % True parameter vector to estimate
x_dim = 3;
y_dim = 1;
N = 1000;           % Number of time samples
M = 100;            % Number of Monte Carlo realizations (reduced for faster testing)
Na = 6;             % Number of agents

% Generate input signals and observations
x = 0.5*randn(1,N+2) + 1;
noise_std = 0.2;
noise = noise_std * randn(1,N,Na,M);
d = zeros(1,N,Na,M);
y = zeros(1,N);

% Generate true observations
for n = 1:N
    y(n) = u * x(n:n+2)';
    d(1,n,:,:) = y(n) + noise(1,n,:,:);
end

%% System Model Setup for Kalman Filters
Q = zeros(x_dim);           % Process noise covariance (stationary system)
A = eye(x_dim);             % State transition matrix (identity for stationary)
R = noise_std^2;            % Measurement noise variance

% Create system model
model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

%% Create Multi-Agent System with Agent2

fprintf('Creating multi-agent system with %d Agent2 instances...\n', Na);

% Create agent vector
agents2_vec = cell(Na, 1);

for a = 1:Na
    % Initial H matrix (can be different for each agent)
    H_matrix_init = [1 0 0];  % Simple initialization
    
    % Create KF_diff technique for each agent
    kf_technique = KF_diff('x_dim', x_dim, 'y_dim', y_dim, ...
                          'H_matrix', H_matrix_init, 'R_matrix', R, ...
                          'Pa_init', {'delta', 0.1}, ...
                          'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                          'system_model', model_sys);
    
    % Create Agent2
    agents2_vec{a} = Agent2('x_dim', x_dim, 'y_dim', y_dim, 'agent_tech', kf_technique);
end

%% Network Topology (B-matrix)
% Define agent connectivity (same as in test_MAS_sim.m)
B_matrix = [1 1 1 0 0 0;
            1 1 0 0 0 0;
            1 0 1 1 0 0;
            0 0 1 1 1 1;
            0 0 0 1 1 0;
            0 0 0 1 0 1];

%% Create Fusion Technique
% For now, use a simple placeholder fusion (to be implemented later)
fusion_technique = Adapt_and_Fuse([], 'fusion_strategy', 'weighted');

%% Monte Carlo Simulation

fprintf('Starting Monte Carlo simulation (%d realizations)...\n', M);

% Initialize result storage
H_hat_history = zeros(y_dim, x_dim, N, Na, M);
y_hat_history = zeros(y_dim, N, Na, M);
xp_hat_history = zeros(x_dim, N, Na, M);  % Posterior state estimates
P_trace_history = zeros(N, Na, M);        % Covariance traces

% Monte Carlo loop
for m = 1:M
    % Reset all agents for new realization
    for a = 1:Na
        agents2_vec{a}.reset();
    end
    
    % Time loop for current realization
    for n = 1:N
        % Self-learning step for all agents
        for a = 1:Na
            agents2_vec{a}.self_learning_step(d(1,n,a,m));
        end
        
        % Social learning step (fusion) - placeholder implementation
        % TODO: Implement proper fusion using Adapt_and_Fuse
        collec_H_hat = zeros(y_dim, x_dim, Na);
        for a = 1:Na
            collec_H_hat(:,:,a) = agents2_vec{a}.get_H_hat();
        end
        
        % Simple consensus fusion for now (to be replaced with Adapt_and_Fuse)
        H_hat_avg = mean(collec_H_hat, 3);
        for a = 1:Na
            agents2_vec{a}.social_learning_step(H_hat_avg);
        end
        
        % Store results
        for a = 1:Na
            H_hat_history(:,:,n,a,m) = agents2_vec{a}.get_H_hat();
            y_hat_history(:,n,a,m) = agents2_vec{a}.get_y_hat();
            xp_hat_history(:,n,a,m) = agents2_vec{a}.get_posterior_state();
            
            % Store covariance trace
            P = agents2_vec{a}.get_posterior_covariance();
            if ~isempty(P)
                P_trace_history(n,a,m) = trace(P);
            end
        end
    end
    
    % Progress indicator
    if mod(m, 10) == 0
        fprintf('Completed %d/%d Monte Carlo realizations\n', m, M);
    end
end

%% Performance Analysis

fprintf('\nAnalyzing performance...\n');

% Select agent for detailed analysis
a_selected = 4;

% Mean Squared Error (MSE) analysis
e_pred = (y_hat_history(1,:,a_selected,:) - d(1,:,a_selected,:)).^2;
e_pred_mean = mean(e_pred, 4);  % Average over Monte Carlo realizations

% Parameter estimation error (MSD - Mean Square Deviation)
e_param = zeros(1,N,M);
for m = 1:M
    for i = 1:N
        e_param(1,i,m) = norm(H_hat_history(:,:,i,a_selected,m) - u)^2;
    end
end
e_param_mean = mean(e_param, 3);

% State estimation error
e_state = zeros(1,N,M);
for m = 1:M
    for i = 1:N
        e_state(1,i,m) = norm(xp_hat_history(:,i,a_selected,m) - u')^2;
    end
end
e_state_mean = mean(e_state, 3);

%% Plotting Results

n = 1:N;

% MSE Plot
figure(1)
plot(n, 10*log10(e_pred_mean), 'b', 'LineWidth', 1.5)
title(sprintf('Multi-Agent Kalman: MSE (Agent %d)', a_selected))
ylabel('MSE [dB]')
xlabel('Time')
ylim([-30 0])
grid on

% MSD Plot
figure(2)
plot(n, 10*log10(e_param_mean), 'r', 'LineWidth', 1.5)
title(sprintf('Multi-Agent Kalman: MSD (Agent %d)', a_selected))
ylabel('MSD [dB]')
xlabel('Time')
ylim([-30 0])
grid on

% State estimation error
figure(3)
plot(n, 10*log10(e_state_mean), 'g', 'LineWidth', 1.5)
title(sprintf('Multi-Agent Kalman: State Estimation Error (Agent %d)', a_selected))
ylabel('Error [dB]')
xlabel('Time')
ylim([-30 0])
grid on

% Covariance evolution
if any(P_trace_history(:) > 0)
    P_trace_mean = mean(P_trace_history(:,a_selected,:), 3);
    figure(4)
    plot(n, P_trace_mean, 'm', 'LineWidth', 1.5)
    title(sprintf('Multi-Agent Kalman: Covariance Trace Evolution (Agent %d)', a_selected))
    ylabel('Trace(P)')
    xlabel('Time')
    grid on
end

%% Performance Metrics

fprintf('\n=== Multi-Agent Kalman Performance Metrics ===\n');
fprintf('Agent analyzed: %d\n', a_selected);
fprintf('Final MSE: %.6f dB\n', 10*log10(e_pred_mean(end)));
fprintf('Final MSD: %.6f dB\n', 10*log10(e_param_mean(end)));
fprintf('Final State Error: %.6f dB\n', 10*log10(e_state_mean(end)));

% Convergence analysis
convergence_threshold_db = -20;  % -20 dB threshold
msd_db = 10*log10(e_param_mean);
convergence_idx = find(msd_db < convergence_threshold_db, 1);
if ~isempty(convergence_idx)
    fprintf('Parameter convergence achieved at sample: %d\n', convergence_idx);
else
    fprintf('Parameter convergence not achieved within threshold %d dB\n', convergence_threshold_db);
end

% Network-wide performance
fprintf('\n=== Network-wide Performance ===\n');
final_errors = zeros(Na, 1);
for a = 1:Na
    agent_error = mean(e_param(1,end-50:end,:), [2,3]);  % Average over last 50 samples and MC realizations
    final_errors(a) = 10*log10(agent_error);
    fprintf('Agent %d final MSD: %.3f dB\n', a, final_errors(a));
end

fprintf('Network average final MSD: %.3f dB\n', mean(final_errors));
fprintf('Network MSD standard deviation: %.3f dB\n', std(final_errors));

fprintf('\nMulti-agent Kalman simulation completed successfully!\n');
