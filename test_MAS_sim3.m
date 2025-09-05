% Test script for Multi-Agent Social Learning with Agent2
% Refactored to work with current codebase implementation
% Uses specified simulation parameters for multi-agent Kalman filtering

clear all; 
% close all;
clc;
global DEBUG_MODE;
DEBUG_MODE = false;
% Add necessary paths
addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")
addpath("./Utils/")

%% Verify Required Classes Exist
fprintf('=== Multi-Agent Social Learning Test (Agent2) ===\n');
fprintf('Verifying required classes...\n');

try
    % Test if required classes can be instantiated
    test_model = Linear_State('dim', 2);
    fprintf('  ✓ Linear_State class available\n');

    % Test KF_diff availability
    if exist('KF_diff', 'class') == 8
        fprintf('  ✓ KF_diff class available\n');
    else
        error('KF_diff class not found');
    end

    % Test Agent2 availability
    if exist('Agent2', 'class') == 8
        fprintf('  ✓ Agent2 class available\n');
    else
        error('Agent2 class not found');
    end

    % Test General_Adapt_and_Fuse availability
    if exist('General_Adapt_and_Fuse', 'class') == 8
        fprintf('  ✓ General_Adapt_and_Fuse class available\n');
    else
        error('General_Adapt_and_Fuse class not found');
    end

catch exception
    fprintf('  ✗ Error: %s\n', exception.message);
    fprintf('Please ensure all required classes are in the MATLAB path.\n');
    return;
end

%% Simulation Setup - Using Specified Parameters

% Simulation parameters (as specified)
x_dim = 3;
y_dim = 1;
y_sd = 1;

u = [1 1 1]';   % The "state"
H = [1 1 1];
N = 200;
M = 50;    % Number of Monte Carlo realizations
Na = 6;  % Number of agents
n = 1:N;
rng(8988466)
noise = y_sd * randn(Na,N,M);

y = H*u;

d = y + noise;

% M = 1;
dims = [N M];

fprintf('Simulation parameters:\n');
fprintf('  State dimension: %d\n', x_dim);
fprintf('  Observation dimension: %d\n', y_dim);
fprintf('  Noise std: %.1f\n', y_sd);
fprintf('  True state: [%s]\n', num2str(u'));
fprintf('  Observation matrix: [%s]\n', num2str(H));
fprintf('  Time steps: %d\n', N);
fprintf('  Monte Carlo realizations: %d\n', M);

%% Multi-Agent System Setup
% Na = 4;  % Number of agents
fprintf('\nCreating %d Agent2 instances...\n', Na);

% Initialize agent storage
agents = cell(Na, 1);

% System model setup (same for all agents)
Q = zeros(x_dim);
A = eye(x_dim);
% model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A); % Verificar, pois acho que está criando a mesma instância para classes distintas, ao passo que, necessariamente, cada filtro KF deve ter o seu próprio system model.

% Create agents with KF_diff technique
for a = 1:Na
    % Use the same H matrix for all agents (constant observation model)
    H_matrix_init = H;  % Use the specified H matrix
    R = y_sd^2;  % Measurement noise variance

    % Create Kalman System Model
    model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

    % Create KF_diff technique
    kf_technique = KF_diff('x_dim', x_dim, 'y_dim', y_dim, ...
                          'H_matrix', H_matrix_init, 'R_matrix', R, ...
                          'Pa_init', {'delta', 0.1}, ...
                          'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                          'system_model', model_sys);

    % Create Agent2 (will use default fusion technique from constructor)
    agents{a} = Agent2('agent_tech', kf_technique);

    fprintf('  Agent %d created with H = [%s]\n', a, num2str(H_matrix_init));
end

%% Network Topology Setup - Configure Fusion Techniques
fprintf('\nConfiguring fusion techniques for distributed neighbor management...\n');

% Define network topology (each agent knows subset of others + itself)
% Agent 1: connected to agents 1,2,3
% Agent 2: connected to agents 1,2,4
% Agent 3: connected to agents 1,3,4
% Agent 4: connected to agents 2,3,4
% neighbor_lists = {
%     [1, 2, 3];     % Agent 1 neighbors (including self)
%     [1, 2, 4];     % Agent 2 neighbors
%     [1, 3, 4];     % Agent 3 neighbors
%     [2, 3, 4]      % Agent 4 neighbors
% };

neighbor_lists = {
    [1, 2, 3];   % Agent 1 neighbors (including self)
    [1, 2];      % Agent 2 neighbors
    [1, 3, 4];   % Agent 3 neighbors
    [3, 4, 5, 6] % Agent 4 neighbors
    [4, 5]       % Agent 5 neighbors
    [4, 6]       % Agent 6 neighbors
};

% Setup fusion techniques and neighbor connections for each agent
for a = 1:Na
    neighbors_idx = neighbor_lists{a};
    n_neighbors = length(neighbors_idx);

    % Equal weighting (including self)
    weights = ones(1, n_neighbors) / n_neighbors;

    % Get neighbor agent objects (preallocate for efficiency)
    neighbor_agents = cell(n_neighbors, 1);
    for i = 1:n_neighbors
        neighbor_agents{i} = agents{neighbors_idx(i)};
    end
    neighbor_agents = [neighbor_agents{:}]';  % Convert to array

    % Create fusion technique for this agent
    fusion_tech = General_Adapt_and_Fuse('neighbors', neighbor_agents, ...
                                        'neighbors_weights', weights);

    % Replace the default fusion technique with configured one
    agents{a}.fusion_technique = fusion_tech;

    fprintf('  Agent %d: %d neighbors (weights: %s)\n', ...
            a, n_neighbors, mat2str(weights, 3));
end

%% Initialize result storage
% H_hat_history = zeros(y_dim, x_dim, N, Na);
y_hat_history = zeros(y_dim, N, Na, M);
xp_hat_history = zeros(x_dim, N, Na, M);  % Posterior state estimates
% xa_hat_history = zeros(x_dim, N, Na);  % Prior state estimates  
P_trace_history = zeros(N, Na, M);        % Covariance traces
individual_estimates = zeros(x_dim, N, Na, M);  % Before fusion
fused_estimates = zeros(x_dim, N, Na, M);       % After fusion

if M>1
    fprintf('\nStarting Monte Carlo simulation with %d time steps (%d realizations)...\n', N, M);
else
    fprintf('\nStarting simulation with %d time steps...\n', N);
end

%% Main Simulation Loop
for m = 1:M    
    for t = 1:N
        % if mod(t, 50) == 0
        %     fprintf('  Processing time step %d/%d\n', t, N);
        % end

        % Step 1: Self-learning step for all agents (individual learning)
        for a = 1:Na
            try
                % Apply individual Kalman filtering
                % [agents{a}.y_hat] = agents{a}.agent_technique.apply('measurement', d(t), ...
                %                                                    'timestamp', datetime('now'));
                % [agents{a}.y_hat] = agents{a}.agent_technique.apply('measurement', d(a,t));

                % Update agent's internal state estimates from Kalman filter
                agents{a}.self_learning_step('measurement', d(a,t,m));
                % agents{a}.xp_hat = agents{a}.agent_technique.xp_hat;
                % agents{a}.xa_hat = agents{a}.agent_technique.xa_hat;

                % Store individual estimates (before fusion)
                individual_estimates(:, t, a, m) = agents{a}.xp_hat;

            catch exception
                fprintf('Error in self_learning_step for agent %d, at time %d, and realization %d: %s\n', a, t, m, exception.message);
                rethrow(exception);
            end
        end

        % Step 2: Social learning step for all agents (fusion step)
        for a = 1:Na
            try
                % Check if fusion technique is properly set
                if isempty(agents{a}.fusion_technique)
                    fprintf('Warning: Agent %d has no fusion technique set, skipping social learning\n', a);
                    fused_estimates(:, t, a, m) = individual_estimates(:, t, a, m);
                    continue;
                end

                % Pass dim as first positional argument (required by current implementation)
                % agents{a}.fusion_technique.social_learning_step(agents{a}, ...
                %     agents{a}.agent_technique.x_dim);
                agents{a}.social_learning_step();

                % Store fused estimates (after fusion)
                % fused_estimates(:, t, a) = agents{a}.xp_hat;
                fused_estimates(:, t, a, m) = agents{a}.fusion_results.state_estimate;

            catch exception
                fprintf('Error in social_learning_step for agent %d, at time %d, and realization %d: %s\n', a, t, m, exception.message);
                % Continue without fusion for this agent
                fused_estimates(:, t, a, m) = individual_estimates(:, t, a, m);
            end
        end
        
        % Step 3: Update agent state estimates based on latest information (this allows a synchro update for all agents)
        for a = 1:Na
            agents{a}.update_agent_estimates();
        end

        % Store results for analysis
        for a = 1:Na
            y_hat_history(:, t, a, m) = agents{a}.y_hat;
            xp_hat_history(:, t, a, m) = agents{a}.xp_hat;
            % xa_hat_history(:, t, a) = agents{a}.xa_hat;
            
            % Get covariance trace
            P = agents{a}.get_posterior_covariance();
            if ~isempty(P)
                P_trace_history(t, a, m) = trace(P);
            end
        end
    end
    % Reset agents for next realization
    for a = 1:Na
        agents{a}.reset();
    end
    % Progress indicator
    if mod(100*m/M, 5) == 0
        fprintf('  Completed %.0f%% of Monte-Carlo realizations\n', 100*m/M);
    end
end
fprintf('Simulation completed successfully!\n');

%% Performance Analysis
fprintf('\n=== Performance Analysis ===\n');

% Calculate errors for each agent
prediction_errors = zeros(N, Na, M);
state_errors_individual = zeros(N, Na, M);
state_errors_fused = zeros(N, Na, M);

for a = 1:Na
    % Prediction errors (comparing predicted observations to true observations)
    % y_pred = squeeze(y_hat_history(1, :, a))';
    % y_pred = squeeze(y_hat_history(1, :, a, :))';
    y_pred = y_hat_history(1, :, a, :);
    prediction_errors(:, a, :) = abs(y_pred - y);

    % State estimation errors (comparing estimated states to true state)
    for m = 1:M
        for t = 1:N
            state_errors_individual(t, a, m) = norm(individual_estimates(:, t, a, m) - u);
            state_errors_fused(t, a, m) = norm(fused_estimates(:, t, a, m) - u);
        end
    end
    state_errors_individual_mean = mean(state_errors_individual, 3);
    state_errors_fused_mean = mean(state_errors_fused, 3);
    fprintf('Agent %d - Final individual error: %.6f, Final fused error: %.6f\n', ...
            a, state_errors_individual_mean(end, a), state_errors_fused_mean(end, a));
end

% Overall performance metrics
% mean_individual_error = mean(state_errors_individual(end, :)); (não faz sentido, pois, referente a cada amostra de tempo, sempre haverá amostras que dão erro maior e outras que dão erro menor, ambas se compensando em média)
% mean_fused_error = mean(state_errors_fused(end, :));
% improvement = (mean_individual_error - mean_fused_error) / mean_individual_error * 100;

mean_individual_error = mean(sqrt(mean(state_errors_individual_mean(:,:).^2, 1))); % TODO: fix the order of the mean to reflect the monte-carlo of the temporal mean
mean_fused_error = mean(sqrt(mean(state_errors_fused_mean(:,:).^2, 1)));
improvement = (mean_individual_error - mean_fused_error) / mean_individual_error * 100;

fprintf('\nOverall Performance:\n');
fprintf('  Total MC and temporal mean individual error: %.6f\n', mean_individual_error);
fprintf('  Total MC and temporal mean fused error: %.6f\n', mean_fused_error);
fprintf('  Improvement from fusion: %.2f%%\n', improvement);

% Additional metrics
fprintf('\nAdditional Metrics:\n');
mc_pred_error = mean(prediction_errors(:, :, :),3);
mean_pred_error = mean(mc_pred_error(end, :), 2); % Mean over agents
fprintf('  MC and mean final prediction error: %.6f\n', mean_pred_error);
fprintf('  True observation value: %.6f\n', y);

%% Visualization
fprintf('\nGenerating plots...\n');
m = 1;
% Figure 1: Observations and Predictions
figure(1);
clf;
subplot(2,2,1);
plot(n, d(1, :, 1, 1), 'b-', 'LineWidth', 1);
hold on;
plot(n, squeeze(y_hat_history(1, :, 1, m)), 'r-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 2, m)), 'g-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 3, m)), 'c-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 4, m)), 'm-', 'LineWidth', 1.5);
plot(n, y*ones(size(n)), 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Observation');
title(sprintf('Observations vs Predictions (realization %d)', m));
legend('Noisy Observations from Agent 1', 'Agent 1 Pred', 'Agent 2 Pred', 'Agent 3 Pred', 'Agent 4 Pred', 'True Value', 'Location', 'best');
grid on;

% Subfigure 2: Consensus Evolution
subplot(2,2,2);
for a = 1:Na
    plot(n, squeeze(fused_estimates(1, :, a, m)), 'LineWidth', 1.5);
    hold on;
end
plot(n, u(1)*ones(size(n)), 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State Component 1');
title(sprintf('Consensus Evolution (x_1), realization %d', m));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'True Value', 'Location', 'best');
grid on;

% Subfigure 3: Covariance Trace Evolution
subplot(2,2,3);
plot(n, P_trace_history(:, 1, m), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, P_trace_history(:, 2, m), 'g-', 'LineWidth', 1.5);
plot(n, P_trace_history(:, 3, m), 'c-', 'LineWidth', 1.5);
plot(n, P_trace_history(:, 4, m), 'm-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Trace(P)');
title(sprintf('Uncertainty Evolution (Covariance Trace) - realization %d', m));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
grid on;

% Subfigure 4: State Convergence
subplot(2,2,4);
plot(n, squeeze(fused_estimates(1, :, 1, m)), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, squeeze(fused_estimates(2, :, 1, m)), 'g-', 'LineWidth', 1.5);
plot(n, squeeze(fused_estimates(3, :, 1, m)), 'c-', 'LineWidth', 1.5);
plot(n, u(1)*ones(size(n)), 'r--', 'LineWidth', 1);
plot(n, u(2)*ones(size(n)), 'g--', 'LineWidth', 1);
plot(n, u(3)*ones(size(n)), 'c--', 'LineWidth', 1);
xlabel('Time Step');
ylabel('State Estimates');
title(sprintf('State Convergence (Agent 1) - realization %d', m));
legend('x_1 estimate', 'x_2 estimate', 'x_3 estimate', 'x_1 true', 'x_2 true', 'x_3 true', 'Location', 'best');
grid on;

% Figure 2: State Individual vs Fused Estimation Errors (firsts 4 agents)
figure(2);
clf;
subplot(2,2,1);
plot(n, state_errors_individual(:, 1, m), 'r--', 'LineWidth', 1);
hold on;
plot(n, state_errors_fused(:, 1, m), 'r-.', 'LineWidth', 2);
plot(n, state_errors_individual(:, 2, m), 'g--', 'LineWidth', 1);
plot(n, state_errors_fused(:, 2, m), 'g-.', 'LineWidth', 2);
plot(n, state_errors_individual(:, 3, m), 'c--', 'LineWidth', 1);
plot(n, state_errors_fused(:, 3, m), 'c-.', 'LineWidth', 2);
plot(n, state_errors_individual(:, 4, m), 'm--', 'LineWidth', 1);
plot(n, state_errors_fused(:, 4, m), 'm-.', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State Estimation Error');
title(sprintf('Individual vs Fused State Errors (first 4 agents) - realization %d', m));
legend('Agent 1 Individual', 'Agent 1 Fused', 'Agent 2 Individual', 'Agent 2 Fused', 'Agent 3 Individual', 'Agent 3 Fused', 'Agent 4 Individual', 'Agent 4 Fused', 'Location', 'best');
grid on;

% subplot(2,2,1);
% for a = 1:Na
%     plot(n, state_errors_individual(:, a), '--', 'LineWidth', 1);
%     hold on;
% end
% for a = 1:Na
%     plot(n, state_errors_fused(:, a), '-.', 'LineWidth', 2);
% end
% xlabel('Time Step');
% ylabel('State Estimation Error');
% title('All Agents: Individual vs Fused Errors');
% legend('Ind 1', 'Ind 2', 'Ind 3', 'Ind 4', 'Fused 1', 'Fused 2', 'Fused 3', 'Fused 4', 'Location', 'best');
% grid on;

% Figure 2: State Individual vs Fused Estimation Errors (firsts 4 agents) in dB
subplot(2,2,2);
plot(n, 10*log10(state_errors_individual(:, 1, m)), 'r-.', 'LineWidth', 1);
hold on;
plot(n, 10*log10(state_errors_fused(:, 1, m)), 'r-', 'LineWidth', 2);
plot(n, 10*log10(state_errors_individual(:, 2, m)), 'g-.', 'LineWidth', 1);
plot(n, 10*log10(state_errors_fused(:, 2, m)), 'g-', 'LineWidth', 2);
plot(n, 10*log10(state_errors_individual(:, 3, m)), 'c-.', 'LineWidth', 1);
plot(n, 10*log10(state_errors_fused(:, 3, m)), 'c-', 'LineWidth', 2);
plot(n, 10*log10(state_errors_individual(:, 4, m)), 'm-.', 'LineWidth', 1);
plot(n, 10*log10(state_errors_fused(:, 4, m)), 'm-', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State Estimation Error');
title(sprintf('Individual vs Fused State Errors [dB] (first 4 agents) - realization %d', m));
legend('Agent 1 Individual', 'Agent 1 Fused', 'Agent 2 Individual', 'Agent 2 Fused', 'Agent 3 Individual', 'Agent 3 Fused', 'Agent 4 Individual', 'Agent 4 Fused', 'Location', 'best');
grid on;


% Figure 2: Prediction Errors
subplot(2,2,3);
% for a = 1:Na
%     plot(n, prediction_errors(:, a), 'LineWidth', 1.5);
%     hold on;
% end
plot(n, prediction_errors(:, 1, m), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, prediction_errors(:, 2, m), 'g-', 'LineWidth', 1.5);
plot(n, prediction_errors(:, 3, m), 'c-', 'LineWidth', 1.5);
plot(n, prediction_errors(:, 4, m), 'm-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors by Agent (first 4 agents) - realization %d', m));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
grid on;

% Figure 2: Prediction Errors [dB]
subplot(2,2,4);
% for a = 1:Na
%     plot(n, prediction_errors(:, a), 'LineWidth', 1.5);
%     hold on;
% end
plot(n, 10*log10(prediction_errors(:, 1, m)), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, 10*log10(prediction_errors(:, 2, m)), 'g-', 'LineWidth', 1.5);
plot(n, 10*log10(prediction_errors(:, 3, m)), 'c-', 'LineWidth', 1.5);
plot(n, 10*log10(prediction_errors(:, 4, m)), 'm-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors by Agent [dB] (first 4 agents) - realization %d', m));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
grid on;

% Figure 3: Final State Estimates
figure(3);
clf;
subplot(1,2,1);
Na_to_show = min(Na, 8);
agents_idx = 1:Na_to_show;
individual_final = state_errors_individual(end, :, m);
fused_final = state_errors_fused(end, :, m);
bar_width = 0.35;
bar(agents_idx - bar_width/2, individual_final(1:Na_to_show), bar_width, 'FaceColor', 'r', 'FaceAlpha', 0.7);
hold on;
bar(agents_idx + bar_width/2, fused_final(1:Na_to_show), bar_width, 'FaceColor', 'b', 'FaceAlpha', 0.7);
if Na > Na_to_show
    xticklabels(1:Na_to_show);
end
xticks(agents_idx);
xlabel('Agent');
ylabel('Final State Error');
if Na > Na_to_show
    title(sprintf('Final Performance Comparison (first %d agents) - realization %d', Na_to_show, m));
else
    title(sprintf('Final Performance Comparison (all %d agents) - realization %d', Na, m));
end
legend('Individual', 'Fused', 'Location', 'best');
grid on;

% Figure 3: Mean State Estimates
subplot(1,2,2);
Na_to_show = min(Na, 8);
agents_idx = 1:Na_to_show;
individual_final = sqrt(mean(state_errors_individual(:, :, m).^2, 1));
fused_final = sqrt(mean(state_errors_fused(:, :, m).^2, 1));
bar_width = 0.35;
bar(agents_idx - bar_width/2, individual_final(1:Na_to_show), bar_width, 'FaceColor', 'r', 'FaceAlpha', 0.7);
hold on;
bar(agents_idx + bar_width/2, fused_final(1:Na_to_show), bar_width, 'FaceColor', 'b', 'FaceAlpha', 0.7);
if Na > Na_to_show
    xticklabels(1:Na_to_show);
end
xticks(agents_idx);
xlabel('Agent');
ylabel('Mean State Error (RMS)');
if Na > Na_to_show
    title(sprintf('Mean Performance Comparison in RMS sense (first %d agents) - realization %d', Na_to_show, m));
else
    title(sprintf('Mean Performance Comparison in RMS sense (all %d agents) - realization %d', Na, m));
end
legend('Individual', 'Fused', 'Location', 'best');
grid on;

fprintf('\n=== Test Summary (realization %d) ===\n', m);
fprintf('Test completed successfully!\n');
fprintf('\nKey findings:\n');
% if improvement > 0
%     fprintf('  - Social learning improved performance by %.2f%%\n', improvement);
% else
%     fprintf('  - Social learning degraded performance by %.2f%%\n', abs(improvement));
% end

% Calculate uncertainty reduction
if any(P_trace_history(1, :, m) > 0)
    uncertainty_reduction = (mean(P_trace_history(1, :, m)) - mean(P_trace_history(end, :, m))) / mean(P_trace_history(1, :, m)) * 100;
    fprintf('  - Average uncertainty reduction (realization %d): %.2f%%\n', m, uncertainty_reduction);
else
    fprintf('  - Uncertainty evolution: covariance traces available\n');
end

% Check consensus
final_states = squeeze(fused_estimates(:, end, :, m));
consensus_variance = var(final_states, 0, 2);
fprintf('  - Final state consensus variance (realization %d): [%.6f %.6f %.6f]\n', m, consensus_variance);

% Convergence analysis
convergence_threshold = 0.1;
converged_agents = sum(state_errors_fused(end, :, m) < convergence_threshold);
fprintf('  - Agents converged (error < %.1f) (realization %d): %d/%d\n', convergence_threshold, m, converged_agents, Na);


%% Visualization Monte-Carlo
fprintf('\nGenerating plots...\n');
% m = 1;
% Figure 1: Observations and Predictions
figure(4);
clf;


% Figure 4: State Convergence
subplot(2,2,4);
plot(n, squeeze(mean(fused_estimates(1, :, 1, :), 4)), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, squeeze(mean(fused_estimates(2, :, 1, :), 4)), 'g-', 'LineWidth', 1.5);
plot(n, squeeze(mean(fused_estimates(3, :, 1, :), 4)), 'c-', 'LineWidth', 1.5);
plot(n, u(1)*ones(size(n)), 'r--', 'LineWidth', 1);
plot(n, u(2)*ones(size(n)), 'g--', 'LineWidth', 1);
plot(n, u(3)*ones(size(n)), 'c--', 'LineWidth', 1);
xlabel('Time Step');
ylabel('State Estimates');
title(sprintf('State Convergence (Agent 1) - Monte-Carlo'));
legend('x_1 estimate', 'x_2 estimate', 'x_3 estimate', 'x_1 true', 'x_2 true', 'x_3 true', 'Location', 'best');
grid on;

% Figure 5: State Individual vs Fused Estimation Errors (firsts 4 agents)
figure(5);
clf;
subplot(2,2,1);
plot(n, mean(state_errors_individual(:, 1, :), 3), 'r--', 'LineWidth', 1);
hold on;
plot(n, mean(state_errors_fused(:, 1, :), 3), 'r-.', 'LineWidth', 2);
plot(n, mean(state_errors_individual(:, 2, :), 3), 'g--', 'LineWidth', 1);
plot(n, mean(state_errors_fused(:, 2, :), 3), 'g-.', 'LineWidth', 2);
plot(n, mean(state_errors_individual(:, 3, :), 3), 'c--', 'LineWidth', 1);
plot(n, mean(state_errors_fused(:, 3, :), 3), 'c-.', 'LineWidth', 2);
plot(n, mean(state_errors_individual(:, 4, :), 3), 'm--', 'LineWidth', 1);
plot(n, mean(state_errors_fused(:, 4, :), 3), 'm-.', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State Estimation Error');
title(sprintf('Individual vs Fused State Errors (first 4 agents) - Monte-Carlo'));
legend('Agent 1 Individual', 'Agent 1 Fused', 'Agent 2 Individual', 'Agent 2 Fused', 'Agent 3 Individual', 'Agent 3 Fused', 'Agent 4 Individual', 'Agent 4 Fused', 'Location', 'best');
grid on;

% subplot(2,2,1);
% for a = 1:Na
%     plot(n, state_errors_individual(:, a), '--', 'LineWidth', 1);
%     hold on;
% end
% for a = 1:Na
%     plot(n, state_errors_fused(:, a), '-.', 'LineWidth', 2);
% end
% xlabel('Time Step');
% ylabel('State Estimation Error');
% title('All Agents: Individual vs Fused Errors');
% legend('Ind 1', 'Ind 2', 'Ind 3', 'Ind 4', 'Fused 1', 'Fused 2', 'Fused 3', 'Fused 4', 'Location', 'best');
% grid on;

% Figure 5: State Individual vs Fused Estimation Errors (firsts 4 agents) in dB
subplot(2,2,2);
plot(n, 10*log10(mean(state_errors_individual(:, 1, :), 3)), 'r-.', 'LineWidth', 1);
hold on;
plot(n, 10*log10(mean(state_errors_fused(:, 1, :), 3)), 'r-', 'LineWidth', 2);
plot(n, 10*log10(mean(state_errors_individual(:, 2, :), 3)), 'g-.', 'LineWidth', 1);
plot(n, 10*log10(mean(state_errors_fused(:, 2, :), 3)), 'g-', 'LineWidth', 2);
plot(n, 10*log10(mean(state_errors_individual(:, 3, :), 3)), 'c-.', 'LineWidth', 1);
plot(n, 10*log10(mean(state_errors_fused(:, 3, :), 3)), 'c-', 'LineWidth', 2);
plot(n, 10*log10(mean(state_errors_individual(:, 4, :), 3)), 'm-.', 'LineWidth', 1);
plot(n, 10*log10(mean(state_errors_fused(:, 4, :), 3)), 'm-', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State Estimation Error');
title(sprintf('Individual vs Fused State Errors [dB] (first 4 agents) - Monte-Carlo'));
legend('Agent 1 Individual', 'Agent 1 Fused', 'Agent 2 Individual', 'Agent 2 Fused', 'Agent 3 Individual', 'Agent 3 Fused', 'Agent 4 Individual', 'Agent 4 Fused', 'Location', 'best');
grid on;


% Figure 5: Prediction Errors
subplot(2,2,3);
% for a = 1:Na
%     plot(n, prediction_errors(:, a), 'LineWidth', 1.5);
%     hold on;
% end
plot(n, mean(prediction_errors(:, 1, :), 3), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, mean(prediction_errors(:, 3, :), 3), 'c-', 'LineWidth', 1.5);
plot(n, mean(prediction_errors(:, 4, :), 3), 'm-', 'LineWidth', 1.5);
plot(n, mean(prediction_errors(:, 2, :), 3), 'g-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors by Agent (first 4 agents) - Monte-Carlo'));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
grid on;

% Figure 5: Prediction Errors [dB]
subplot(2,2,4);
% for a = 1:Na
%     plot(n, prediction_errors(:, a), 'LineWidth', 1.5);
%     hold on;
% end
plot(n, 10*log10(mean(prediction_errors(:, 1, :), 3)), 'r-', 'LineWidth', 1.5);
hold on;
plot(n, 10*log10(mean(prediction_errors(:, 2, :), 3)), 'g-', 'LineWidth', 1.5);
plot(n, 10*log10(mean(prediction_errors(:, 3, :), 3)), 'c-', 'LineWidth', 1.5);
plot(n, 10*log10(mean(prediction_errors(:, 4, :), 3)), 'm-', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors by Agent [dB] (first 4 agents) - Monte-Carlo'));
legend('Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Location', 'best');
grid on;

% Figure 6: Final State Estimates
figure(6);
clf;
subplot(1,2,1);
Na_to_show = min(Na, 8);
agents_idx = 1:Na_to_show;
individual_final = mean(state_errors_individual(end, :, :), 3);
fused_final = mean(state_errors_fused(end, :, :), 3);
bar_width = 0.35;

bar(agents_idx - bar_width/2, individual_final(1:Na_to_show), bar_width, 'FaceColor', 'r', 'FaceAlpha', 0.7);
hold on;
bar(agents_idx + bar_width/2, fused_final(1:Na_to_show), bar_width, 'FaceColor', 'b', 'FaceAlpha', 0.7);
if Na > Na_to_show
    xticklabels(1:Na_to_show);
end
xticks(agents_idx);
xlabel('Agent');
ylabel('Final State Error');
if Na > Na_to_show
    title(sprintf('Final Performance Comparison (first %d agents) - Monte-Carlo', Na_to_show));
else
    title(sprintf('Final Performance Comparison (all %d agents) - Monte-Carlo', Na));
end
legend('Individual', 'Fused', 'Location', 'best');
grid on;

% Figure 6: Mean State Estimates
subplot(1,2,2);
Na_to_show = min(Na, 8);
agents_idx = 1:Na_to_show;
individual_final = sqrt(mean(mean(state_errors_individual(:, :, :), 3).^2, 1));
fused_final = sqrt(mean(mean(state_errors_fused(:, :, :), 3).^2, 1));
bar_width = 0.35;
bar(agents_idx - bar_width/2, individual_final(1:Na_to_show), bar_width, 'FaceColor', 'r', 'FaceAlpha', 0.7);
hold on;
bar(agents_idx + bar_width/2, fused_final(1:Na_to_show), bar_width, 'FaceColor', 'b', 'FaceAlpha', 0.7);
if Na > Na_to_show
    xticklabels(1:Na_to_show);
end
xticks(agents_idx);
xlabel('Agent');
ylabel('Mean State Error (RMS)');
if Na > Na_to_show
    title(sprintf('Mean Performance Comparison in RMS sense (first %d agents) - Monte-Carlo', Na_to_show));
else
    title(sprintf('Mean Performance Comparison in RMS sense (all %d agents) - Monte-Carlo', Na));
end
legend('Individual', 'Fused', 'Location', 'best');
grid on;

fprintf('\n=== Test Summary (Monte-Carlo) ===\n');
fprintf('Test completed successfully!\n');
fprintf('\nKey findings:\n');
if improvement > 0
    fprintf('  - Social learning improved performance by %.2f%%\n', improvement);
else
    fprintf('  - Social learning degraded performance by %.2f%%\n', abs(improvement));
end

% Calculate uncertainty reduction
if any(P_trace_history(1, :, m) > 0)
    uncertainty_reduction = (mean(mean(P_trace_history(1, :, :),3)) - mean(mean(P_trace_history(end, :, :),3))) / mean(mean(P_trace_history(1, :, :),3)) * 100;
    fprintf('  - Average uncertainty reduction (Monte-Carlo): %.2f%%\n', uncertainty_reduction);
else
    fprintf('  - Uncertainty evolution: covariance traces available\n');
end

% Check consensus
final_states = squeeze(mean(fused_estimates(:, end, :, :), 4));
consensus_variance = var(final_states, 0, 2);
fprintf('  - Final state consensus variance (Monte-Carlo): [%.6f %.6f %.6f]\n', consensus_variance);

% Convergence analysis
convergence_threshold = 0.1;
converged_agents = sum(mean(state_errors_fused(end, :, :), 3) < convergence_threshold);
fprintf('  - Agents converged (error < %.1f) (Monte-Carlo): %d/%d\n', convergence_threshold, converged_agents, Na);

fprintf('\nSimulation validated multi-agent social learning with:\n');
fprintf('  - Constant observation model H = [%s]\n', num2str(H));
fprintf('  - True state u = [%s]\n', num2str(u'));
fprintf('  - Distributed fusion using General_Adapt_and_Fuse\n');
fprintf('  - %d agents with network topology\n', Na);
