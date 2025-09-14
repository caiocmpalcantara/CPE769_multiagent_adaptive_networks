% Test script for Multi-Agent Social Learning with Agent
% Refactored to work with current codebase implementation
% Uses specified simulation parameters for multi-agent Kalman filtering

% clear all; 
% close all;
% clc;
global DEBUG_MODE;
DEBUG_MODE = false;

% Add necessary paths
addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")
addpath("./Utils/")

diary log.txt; % Open diary file for writing
diary on
%% Verify Required Classes Exist
fprintf('=== Multi-Agent Social Learning Test (Agent) ===\n');
fprintf('Verifying required classes...\n');

try
    % Test if required classes can be instantiated
    test_model = Linear_State('dim', 2);
    fprintf('  ✓ Linear_State class available\n');

    % Test KF availability
    if exist('KF', 'class') == 8
        fprintf('  ✓ KF class available\n');
    else
        error('KF class not found');
    end

    % Test RLS availability
    if exist('RLS', 'class') == 8
        fprintf('  ✓ RLS class available\n');
    else
        error('RLS class not found');
    end

    % Test Agent availability
    if exist('Agent', 'class') == 8
        fprintf('  ✓ Agent class available\n');
    else
        error('Agent class not found');
    end

    % Test General_Adapt_and_Fuse availability
    if exist('General_Adapt_and_Fuse', 'class') == 8
        fprintf('  ✓ General_Adapt_and_Fuse class available\n');
    else
        error('General_Adapt_and_Fuse class not found');
    end

    % Test Diff_KF_time_measure availability
    if exist('Diff_KF_time_measure', 'class') == 8
        fprintf('  ✓ Diff_KF_time_measure class available\n');
    else
        error('Diff_KF_time_measure class not found');
    end

    % Test Diff_KF_info_matrix availability
    if exist('Diff_KF_info_matrix', 'class') == 8
        fprintf('  ✓ Diff_KF_info_matrix class available\n');
    else
        error('Diff_KF_info_matrix class not found');
    end

catch exception
    fprintf('  ✗ Error: %s\n', exception.message);
    fprintf('Please ensure all required classes are in the MATLAB path.\n');
    return;
end

%% Simulation Setup - Using Specified Parameters
% @brief Need to select the network topology, the technique and the fusion technique
% @arguments: , technique, fusion_tech
%   - net_topology: Network topology (neighbor lists)
%   - tech: Adaptive filtering technique that will be used by Agents
%   - fusion_tech: Fusion technique that will be used by Agents 
% @options: 
%   - net_topology: 'caio_net_topology', 'merched_net_topology'
%   - tech: 'KF', 'RLS'
%   - str_fusion_tech: 'General_Adapt_and_Fuse', 'Diff_KF_time_measure', 'Diff_KF_info_matrix' or '' none ➜ Non-Cooperative

% Simulation parameters (as specified)
x_dim = 3;  % State dimension
y_dim = 1;  % Observation dimension
N = 500;     % Time steps
M = 50;      % Number of Monte Carlo realizations

% Define network topology (each agent knows subset of others + itself)
switch net_topology
    case 'caio_net_topology'
        neighbor_lists = { % all including self
            [1, 2, 3];      % Agent 1 neighbors 
            [1, 2];         % Agent 2 neighbors
            [1, 3, 4];      % Agent 3 neighbors
            [3, 4, 5, 6];   % Agent 4 neighbors
            [4, 5];         % Agent 5 neighbors
            [4, 6];         % Agent 6 neighbors
        };
    case 'merched_net_topology'
        neighbor_lists = { % all including self
            [1, 4, 14];                 % Agent 1 neighbors
            [2, 6, 8];                  % Agent 2 neighbors
            [3, 4, 5, 13, 15];          % Agent 3 neighbors
            [1, 3, 4, 15];              % Agent 4 neighbors
            [4, 5, 10, 14, 15];         % Agent 5 neighbors
            [6, 8, 9];                  % Agent 6 neighbors
            [7, 12, 14, 18];            % Agent 7 neighbors
            [2, 8, 9, 10];              % Agent 8 neighbors
            [2, 6, 8, 9];               % Agent 9 neighbors
            [5, 10, 13, 17];            % Agent 10 neighbors
            [11, 17];                   % Agent 11 neighbors
            [12, 14, 18];               % Agent 12 neighbors
            [3, 5, 13];                 % Agent 13 neighbors
            [5, 7, 12, 14, 15, 20];     % Agent 14 neighbors
            [1, 3, 5, 14, 15];          % Agent 15 neighbors
            [10, 16];                   % Agent 16 neighbors
            [10, 16, 17];               % Agent 17 neighbors
            [7, 14, 18, 20];            % Agent 18 neighbors
            [3, 19];                    % Agent 19 neighbors
            [7, 18, 20];                % Agent 20 neighbors
        };
        
    otherwise % simple case
        neighbor_lists = { % all including self
            [1, 2, 3];     % Agent 1 neighbors
            [1, 2, 4];     % Agent 2 neighbors
            [1, 3, 4];     % Agent 3 neighbors
            [2, 3, 4]      % Agent 4 neighbors
        };
        warning(strcat('Network topology not implemented. Using simple case. Possible options: ', ...
                                                                    '''caio_net_topology''', ...
                                                                    '''merched_net_topology'''));
end

figure(10)
print_graph(neighbor_lists);

Na = length(neighbor_lists);  % Number of agents

% Assumption: same state dynamic for all agents, different noise observations
w = zeros(3,1);
w(:,1) = [-0.2 0.7 0.3]';   % The initial state

noisePowers_dB = [ ...
  -27.6, -24.2, -10.3, -22.4, -26.6, ...
  -17.1, -23.1, -21.7, -21.2, -25.5, ...
  -13.3, -21.6, -25.7, -20.0, -10.4, ...
  -15.7, -20.4, -11.6, -20.9, -24.7 ];

% Autoregressive model input (same to all agents)

rk = 0.95;
rng(8988466)
% Defining the excitation signal variance
switch net_topology      
    case 'merched_net_topology'
        regressionPower_dB = [ ...
            12.0, 10.4, 12.5, 12.5, 10.0, ...
            12.6, 12.3, 12.2, 12.4, 11.5, ...
            11.6, 11.4, 12.6, 12.6, 12.5, ...
            10.4, 11.5, 12.1, 10.4, 12.2 ];
        trace_I_M = 10.^(regressionPower_dB(1:Na)/10);
        u_sd = sqrt(trace_I_M/x_dim);

        x = zeros(x_dim,N,Na);
        rng(8988466)
        for a = 1:Na
            x(:,:,a) = u_sd(a) .* randn(3,N); % excitation
        end
        u = zeros(x_dim,N,Na); % Kalman => H
        for i = 2:N
            u(:,i,a) = rk * u(:,i-1,a)  + sqrt(1-rk^2) * x(:,i,a);
        end

        d = zeros(1,Na,N,M);
        H = zeros(N,3,Na);
        y = zeros(N,Na);
        for n = 1:N
            for a = 1:Na
                H(n,:,a) = u(:,n,a)';
                y(n,a) = H(n,:,a)*w; % TODO: Refactor the y and H indexes in simulations below
                d(:,a,n,:) = y(n,a) + 10^(noisePowers_dB(a)/10) * randn(1,1,1,M);
            end
        end

    otherwise
        u_sd = .5;
        rng(8988466)
        x = u_sd * randn(3,N); % excitation
        u = zeros(x_dim,N); % Kalman => H
        d = zeros(1,Na,N,M);
        H = zeros(N,3,Na);
        y = zeros(N,Na);
        for n = 2:N
            u(:,n) = rk * u(:,n-1)  + sqrt(1-rk^2) * x(:,n);
            H(n,:,1) = u(:,n)';
            y(n,:) = H(n,:,1)*w;
            for a = 1:Na
                d(:,a,n,:) = y(n,1) + 10^(noisePowers_dB(a)/10) * randn(1,1,1,M);
                H(n,:,a) = H(n,:,1);
            end
        end
end

dims = [N M];

fprintf('Simulation parameters:\n');
fprintf('  State dimension: %d\n', x_dim);
fprintf('  Observation dimension: %d\n', y_dim);
fprintf('  Noise std (per agent): [%s]\n', num2str(10.^(noisePowers_dB(1:Na)/10)));
if strcmp(net_topology, 'merched_net_topology')
    fprintf('  Regression power (per agent): [%s]\n', num2str(trace_I_M));
    fprintf('  SNR (per agent) [dB]: [%s]\n', num2str(10*log10(trace_I_M./10.^(noisePowers_dB(1:Na)/10))));
else
    fprintf('  Regression power (per agent): [%s]\n', num2str(x_dim*u_sd^2));
    fprintf('  SNR (per agent) [dB]: [%s]\n', num2str(10*log10(x_dim*u_sd^2./10.^(noisePowers_dB(1:Na)/10))));
end
fprintf('  True state: [%s]\n', num2str(w'));
% fprintf('  Observation matrix: [%s]\n', num2str(H(n,:)));
fprintf('  Time steps: %d\n', N);
fprintf('  Monte Carlo realizations: %d\n', M);


%% Initialize the variables to the Monte-Carlo result storage

y_hat_history = zeros(y_dim, N, Na, M);
xp_hat_history = zeros(x_dim, N, Na, M);        % Posterior state estimates
P_trace_history = zeros(N, Na, M);              % Covariance traces
individual_estimates = zeros(x_dim, N, Na, M);  % Before fusion
fused_estimates = zeros(x_dim, N, Na, M);       % After fusion
R = zeros(Na,M);
agents = cell(Na, M);                           % Initialize agent storage for parallelization


% System model setup (same for all agents)
Q = zeros(x_dim);
A = eye(x_dim);

noisePowers_dB = repmat(noisePowers_dB(1:Na), [Na, M]);
H = repmat(H,[1,1,1,M]);

% for a = 1:Na
%     neighbors_idx = neighbor_lists{a};
%     n_neighbors = length(neighbors_idx);

%     % Equal weighting (including self)
%     weights = ones(1, n_neighbors) / n_neighbors;

%     % Get neighbor agent objects (preallocate for efficiency)
%     neighbor_agents = cell(n_neighbors, 1);
%     for i = 1:n_neighbors
%         neighbor_agents{i} = agents{neighbors_idx(i)};
%     end
%     neighbor_agents = [neighbor_agents{:}]';  % Convert to array
% end

%% Multi-Agent System Setup
fprintf('\n======================================================\n')
if M>1
    fprintf('Starting Monte Carlo Multi-Agent System setup with %d time steps (%d realizations)...\n', N, M);
else
    fprintf('Starting Multi-Agent System setup with %d time steps...\n', N);
end
for m = 1:M

    fprintf('\nCreating %d Agent instances per Monte Carlo realization %d....\n', Na, m);
    % Agent Technique Setup - Configure Agent techniques
    %    Create agents with Agent technique
    for a = 1:Na
        % Use the same H matrix for all agents (constant observation model)
        H_matrix_init = H(1,:,a,m);  % Use the specified H matrix
        R(a,m) = (10^(noisePowers_dB(a,m)/10))^2;  % Measurement noise variance

        % Create Kalman System Model
        model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

        % Create agent technique
        switch tech
            case 'KF'
                agent_technique = KF('x_dim', x_dim, 'y_dim', y_dim, ...
                                    'H_matrix', H_matrix_init, 'R_matrix', R(a,m), ...
                                    'Pa_init', {'delta', 0.1}, ...
                                    'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                                    'system_model', model_sys);
            case 'RLS'
                start_vals = struct('delta', 0.1, 'initial_state', zeros(x_dim, 1));
                agent_technique = RLS('x_dim', x_dim, 'y_dim', y_dim, ...
                                'H_matrix', H_matrix_init, ...
                                'lambda', 0.98, ...
                                'start_vals', start_vals);

            otherwise
                error('Unsupported technique: %s', tech);
        end

        
        % Create Agent (will use default fusion technique from constructor)
        agents{a,m} = Agent('agent_tech', agent_technique);
        fprintf('  Agent %d created, realization %d\n', a, m);
    end

    fprintf('\nConfiguring fusion techniques for distributed neighbor management per Monte Carlo realization %d....\n', Na, m);
    % Network Topology Setup - Configure Fusion Techniques
    %   Setup fusion techniques and neighbor connections for each agent
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
        switch str_fusion_tech
            case 'General_Adapt_and_Fuse'
                fusion_tech = General_Adapt_and_Fuse('neighbors', neighbor_agents, ...
                                            'neighbors_weights', weights);
            case 'Diff_KF_time_measure'
                if isa(agents{a,m}.agent_technique, 'KF')
                    fusion_tech = Diff_KF_time_measure('neighbors', neighbor_agents, ...
                                            'neighbors_weights', weights);
                end
            case 'Diff_KF_info_matrix'
                if isa(agents{a,m}.agent_technique, 'KF')
                    fusion_tech = Diff_KF_info_matrix('neighbors', neighbor_agents, ...
                                            'neighbors_weights', weights);
                end
            otherwise
                warning('Unsupported fusion technique: %s\n Selecting Noon_coop.', str_fusion_tech);
                fusion_tech = [];
        end

        % Replace the default fusion technique with configured one
        if ~isempty(fusion_tech)
            agents{a,m}.fusion_technique = fusion_tech;
        end
        
        fprintf('  Agent %d: %d neighbors (weights: %s), realization %d\n', ...
                a, n_neighbors, mat2str(weights, 3), m);
    end
end

%% Main Simulation Loop
fprintf('\n======================================================\n')
if M>1
    fprintf('\nStarting Monte Carlo simulation with %d time steps (%d realizations)...\n', N, M);
else
    fprintf('\nStarting simulation with %d time steps...\n', N);
end

parfor m = 1:M    
    for t = 1:N
        % if mod(t, 50) == 0
        %     fprintf('  Processing time step %d/%d\n', t, N);
        % end
        DEBUG(sprintf('  Processing time step %d/%d in realization %d...\n', t, N, m));
        % Step 1: Self-learning step for all agents (individual learning)
        DEBUG(sprintf('    Step 1: Self-learning step for all agents (individual learning), realization %d...\n', m));
        for a = 1:Na
            try
                % Update agent's internal state estimates from Kalman filter
                agents{a,m}.self_learning_step('measurement', d(1,a,t,m), 'H_matrix', H(t,:,a,m));

                % Store individual estimates (before fusion)
                individual_estimates(:, t, a, m) = agents{a,m}.xp_hat;
                % fprintf('Individual estimate for agent %d at time %d: %s\n', a, t, mat2str(individual_estimates(:, t, a, m)));

            catch exception
                % fprintf('Error in self_learning_step for agent %d, at time %d, and realization %d: %s\n', a, t, m, exception.message);
                rethrow(exception);
            end
        end

        % % Step 1.1: Update agent state estimates based on latest information (this allows a synchro update for all agents) - Needed for diffusion techniques in Social_Learning in KF (Diff_KF_time_measure)
        if ismethod(agents{a,m}.fusion_technique, 'apply_incremental_step') % Fusion techniques that need an incremental step before the fusion step
            DEBUG(sprintf('    Step 1.1: Update agent state estimates for diffusion techniques with an incremental step, realization %d...\n', m));
            for a = 1:Na
                agents{a,m}.fusion_technique.apply_incremental_step('self_agent', agents{a,m}, 'y_dim', y_dim);
            end
        end

        % Step 2: Social learning step for all agents (fusion step)
        DEBUG(sprintf('    Step 2: Social learning step for all agents (fusion step), realization %d...\n', m));
        for a = 1:Na
            try
                % Check if fusion technique is properly set
                agents{a,m}.social_learning_step();
                if isa(agents{a,m}.fusion_technique, 'Noon_coop')
                    % fprintf('Warning: Agent %d has no fusion technique set, skipping social learning\n', a);
                    fused_estimates(:, t, a, m) = individual_estimates(:, t, a, m);
                    continue;
                end
                
                fused_estimates(:, t, a, m) = agents{a,m}.fusion_results.state_estimate;
                % fprintf('Fused estimate for agent %d at time %d: %s\n', a, t, mat2str(fused_estimates(:, t, a, m)'));
                              

            catch exception
                fprintf('Error in social_learning_step for agent %d, at time %d, and realization %d: %s\n', a, t, m, exception.message);
                % Continue without fusion for this agent
                fused_estimates(:, t, a, m) = individual_estimates(:, t, a, m);
            end
        end
        
        % Step 3: Update agent state estimates based on latest information (this allows a synchro update for all agents)
        DEBUG(sprintf('    Step 3: Update agent state estimates based on latest information, realization %d...\n', m));
        for a = 1:Na
            agents{a,m}.update_agent_estimates();
        end

        % Store results for analysis
        for a = 1:Na
            y_hat_history(:, t, a, m) = agents{a,m}.y_hat;
            xp_hat_history(:, t, a, m) = agents{a,m}.xp_hat;
            
            % Get covariance trace
            P = agents{a,m}.get_posterior_covariance();
            if ~isempty(P)
                P_trace_history(t, a, m) = trace(P);
            end
        end
    end
    
    % % Reset agents for next realization
    % for a = 1:Na
    %     agents{a,m}.reset();
    % end
    % % Progress indicator
    % if mod(100*m/M, 5) == 0 && M>1
    %     fprintf('  Completed %.0f%% of Monte-Carlo realizations\n', 100*m/M);
    % end
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
    y_pred = reshape(y_pred, [N, M]);
    for m = 1:M
        for t = 1:N
            prediction_errors(t, a, m) = abs(y_pred(t, m) - y(t,a));
        end
    end

    % State estimation errors (comparing estimated states to true state)
    for m = 1:M
        for t = 1:N
            state_errors_individual(t, a, m) = norm(individual_estimates(:, t, a, m) - w);
            state_errors_fused(t, a, m) = norm(fused_estimates(:, t, a, m) - w);
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
% fprintf('  True observation value: %.6f\n', y(end));

%% Visualization
fprintf('\nGenerating plots...\n');
m = 1;
% Figure 1: Observations and Predictions
figure(1);
clf;
n = 1:N;
subplot(2,2,1);
plot(n, squeeze(d(1, 1, :, 1)), 'b-', 'LineWidth', 1);
hold on;
plot(n, squeeze(y_hat_history(1, :, 1, m)), 'r-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 2, m)), 'g-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 3, m)), 'c-', 'LineWidth', 1.5);
plot(n, squeeze(y_hat_history(1, :, 4, m)), 'm-', 'LineWidth', 1.5);
plot(n, y, 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Observation');
title(sprintf('Observations vs Predictions (realization %d)', m));
legend('Noisy Observations from Agent 1', 'Agent 1 Pred', 'Agent 2 Pred', 'Agent 3 Pred', 'Agent 4 Pred', 'True Value', 'Location', 'best');
grid on;

figure(2)
clf;
% Subfigure 2: Consensus Evolution
subplot(2,2,2);
for a = 1:Na
    plot(n, squeeze(fused_estimates(1, :, a, m)), 'LineWidth', 1.5);
    hold on;
end
plot(n, w(1)*ones(size(n)), 'k--', 'LineWidth', 2);
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
plot(n, w(1)*ones(size(n)), 'r--', 'LineWidth', 1);
plot(n, w(2)*ones(size(n)), 'g--', 'LineWidth', 1);
plot(n, w(3)*ones(size(n)), 'c--', 'LineWidth', 1);
xlabel('Time Step');
ylabel('State Estimates');
title(sprintf('State Convergence (Agent 1) - realization %d', m));
legend('x_1 estimate', 'x_2 estimate', 'x_3 estimate', 'x_1 true', 'x_2 true', 'x_3 true', 'Location', 'best');
grid on;

% Figure 2: State Individual vs Fused Estimation Errors (firsts 4 agents)
figure(3);
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
figure(4);
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
if improvement > 0
    fprintf('  - Social learning improved performance by %.2f%%\n', improvement);
else
    fprintf('  - Social learning degraded performance by %.2f%%\n', abs(improvement));
end

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
if M>1    
    fprintf('\nGenerating Monte-Carlo averaged plots...\n');
    % m = 1;
    % Figure 1: Observations and Predictions
    figure(5);
    clf;


    % Figure 4: State Convergence
    subplot(2,2,4);
    plot(n, squeeze(mean(fused_estimates(1, :, 1, :), 4)), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(n, squeeze(mean(fused_estimates(2, :, 1, :), 4)), 'g-', 'LineWidth', 1.5);
    plot(n, squeeze(mean(fused_estimates(3, :, 1, :), 4)), 'c-', 'LineWidth', 1.5);
    plot(n, w(1)*ones(size(n)), 'r--', 'LineWidth', 1);
    plot(n, w(2)*ones(size(n)), 'g--', 'LineWidth', 1);
    plot(n, w(3)*ones(size(n)), 'c--', 'LineWidth', 1);
    xlabel('Time Step');
    ylabel('State Estimates');
    title(sprintf('State Convergence (Agent 1) - Monte-Carlo'));
    legend('x_1 estimate', 'x_2 estimate', 'x_3 estimate', 'x_1 true', 'x_2 true', 'x_3 true', 'Location', 'best');
    grid on;

    % Figure 5: State Individual vs Fused Estimation Errors (firsts 4 agents)
    figure(6);
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
    figure(7);
    clf;
    title('Final State Estimates - Monte-Carlo');
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
    ylabel('Final RMSD');
    if Na > Na_to_show
        title(sprintf('Final Performance Comparison (RMSD) (first %d agents)', Na_to_show));
    else
        title(sprintf('Final Performance Comparison (RMSD) (all %d agents)', Na));
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
    ylabel('Overtime Mean RMSD');
    if Na > Na_to_show
        title(sprintf('Mean Performance Comparison (RMSD) (first %d agents)', Na_to_show));
    else
        title(sprintf('Mean Performance Comparison (RMSD) (all %d agents)', Na));
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

end

figure(8);
clf;
final_msd_vec =  mean(mean(state_errors_fused, 3),2);
plot(n, 20*log10(final_msd_vec), 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Final MSD');
if M > 1
    title(sprintf('Final MSD - Monte-Carlo: %d realizations', M));
else
    title('Final MSD');
end
grid on


fprintf('\n=====================================================\n');
fprintf('Simulation validated multi-agent social learning with:\n');
% fprintf('  - Constant observation model H = [%s]\n', num2str(H(t,:)));
fprintf('  - True state w = [%s]\n', num2str(w'));
fprintf('  - Agent technique:  %s\n', class(agents{1}.agent_technique));
fprintf('  - Distributed fusion: %s\n', class(agents{1}.fusion_technique));
fprintf('  - %d agents with network topology\n', Na);

diary off;
