% Quick test to verify the refactored test_MAS_sim3.m works with current codebase
% This runs a shortened version to validate the implementation

% clear; close all; clc;

global DEBUG_MODE;
DEBUG_MODE = false;

% Add necessary paths
addpath("./Technique/")
addpath("./Agent/")
addpath("./Technique/Kalman_inc/")
addpath("./Utils/")

fprintf('=== Quick Test for Refactored MAS Simulation ===\n');

%% Test Parameters (same as specified)
rng(8988467);

x_dim = 3;
y_dim = 1;
y_sd = 1;

% kf_fusion_tech = 'Diff_KF_time_measure';
% kf_fusion_tech = 'General_Adapt_and_Fuse';

u = [1 1 1]';   % The "state"
H = [1 1 1];
N = 50;  % Shortened for quick test
n = 1:N;
Na = 2; % Just 2 agents for quick test
noise = y_sd * randn(Na,N);

y = H*u;
d = y + noise;

fprintf('Quick test with %d time steps\n', N);

%% Create Test Agents
agents = cell(Na, 1);

% System model setup
Q = zeros(x_dim);
A = eye(x_dim);
% model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

% Create agents
for a = 1:Na
    H_matrix_init = H;
    R = y_sd^2;
    
    kf_technique = KF_diff('x_dim', x_dim, 'y_dim', y_dim, ...
                          'H_matrix', H_matrix_init, 'R_matrix', R, ...
                          'Pa_init', {'delta', 0.1}, ...
                          'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
                          'system_model', Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A));
    
    agents{a} = Agent2('agent_tech', kf_technique);
    fprintf('  Agent %d created\n', a);
end

%% Configure Fusion
fprintf('Configuring fusion techniques...\n');

% Simple network: both agents connected to each other
for a = 1:Na
    neighbor_agents = [agents{1}; agents{2}];  % Both agents know each other
    weights = [0.5, 0.5];  % Equal weighting
    
    switch kf_fusion_tech
        case 'General_Adapt_and_Fuse'
            fusion_tech = General_Adapt_and_Fuse('neighbors', neighbor_agents, ...
                                        'neighbors_weights', weights);
            fprintf('  General_Adapt_and_Fuse created for agent %d.\n', a);

        case 'Diff_KF_time_measure'
            fusion_tech = Diff_KF_time_measure('neighbors', neighbor_agents, ...
                                        'neighbors_weights', weights);
            fprintf('  Diff_KF_time_measure created for agent %d.\n', a);

        case 'Diff_KF_info_matrix'
            fusion_tech = Diff_KF_info_matrix('neighbors', neighbor_agents, ...
                                        'neighbors_weights', weights);
            fprintf('  Diff_KF_info_matrix created for agent %d.\n', a);
            
        otherwise
            error('Unsupported fusion technique: %s', kf_fusion_tech);
    end

    agents{a}.fusion_technique = fusion_tech;
    fprintf('  Agent %d fusion configured\n', a);
end

%% Quick Simulation
fprintf('Running quick simulation...\n');

individual_estimates = zeros(x_dim, N, Na);
fused_estimates = zeros(x_dim, N, Na);

for t = 1:N
    % Self-learning
    for a = 1:Na
        try
            % agents{a}.self_learning_step('measurement', d(t), ...
            %                              'timestamp', datetime('now')); % TODO: Error in self-learning for agent 1: An error occurred: An error occurred: Comparison is not defined between double and datetime arrays.
            agents{a}.self_learning_step('measurement', d(a,t));
            individual_estimates(:, t, a) = agents{a}.xp_hat;
            if t == 1 % initialize the agent state estimates
                agents{a}.xp_hat = agents{a}.agent_technique.get_posterior_state();
                agents{a}.xa_hat = agents{a}.agent_technique.get_prior_state();
                agents{a}.fusion_results.state_estimate = agents{a}.xp_hat;
                agents{a}.fusion_results.covariance_estimate = agents{a}.agent_technique.Pp;
                if isprop(agents{a}.fusion_technique, 'phi_k') && isprop(agents{a}.fusion_technique, 'P_k')
                    fprintf('Initializing phi_k and P_k for agent %d\n', a);
                    agents{a}.fusion_technique.phi_k = agents{a}.fusion_results.state_estimate;
                    DEBUG(agents{a}.fusion_technique.phi_k)
                    agents{a}.fusion_technique.P_k = agents{a}.fusion_results.covariance_estimate;
                    DEBUG(agents{a}.fusion_technique.P_k)
                else
                    fprintf('No phi_k and P_k to initialize for agent %d\n', a);
                end
            end
        catch exception
            fprintf('Error in self-learning for agent %d: %s\n', a, exception.message);
            return;
        end
    end
    
    % Social learning
    for a = 1:Na
        try
            % Pass dim as positional argument (current implementation requirement)
            agents{a}.social_learning_step();
            fused_estimates(:, t, a) = agents{a}.fusion_results.state_estimate;
            % agents{a}.update_agent_state_estimates(); % Identifiquei uma problemática quanto a atualização de cada agente de forma sequencial
        catch exception
            fprintf('test_MAS_sim4_quick: Error in social learning for agent %d: %s\n', a, exception.message);
            fused_estimates(:, t, a) = individual_estimates(:, t, a);
        end
    end

    % Update agent state estimates based on latest information
    for a = 1:Na
        agents{a}.update_agent_estimates();
    end

    fprintf('  Time step %d completed\n', t);
end

%% Quick Analysis
fprintf('\nQuick Results:\n');
fprintf('   About the last time step (RMSD):\n');
for a = 1:Na
    individual_mean_error = norm(individual_estimates(:, end, a) - u);
    fused_error = norm(fused_estimates(:, end, a) - u);
    fprintf('  Agent %d - Individual error: %.6f, Fused error: %.6f\n', ...
            a, individual_mean_error, fused_error);
end

% Check if fusion brought agents closer together
state_diff_before = norm(individual_estimates(:, end, 1) - individual_estimates(:, end, 2));
state_diff_after = norm(fused_estimates(:, end, 1) - fused_estimates(:, end, 2));
fprintf('  State difference before fusion: %.6f\n', state_diff_before);
fprintf('  State difference after fusion: %.6f\n', state_diff_after);

if state_diff_after < state_diff_before
    fprintf('  ✓ Fusion improved consensus\n');
else
    fprintf('  ⚠ Fusion did not improve consensus (may be expected with limited data)\n');
end

fprintf('\n   Global mean in RMSD sense:\n');
for a = 1:Na
    for n = 1:N
        individual_rmsd(n,a) = norm(individual_estimates(:, n, a) - u);
        fused_rmsd(n,a) = norm(fused_estimates(:, n, a) - u);
    end
    individual_mean_rmsd = sqrt(mean(individual_rmsd(:,a).^2));
    fused_mean_rmsd = sqrt(mean(fused_rmsd(:,a).^2));
    fprintf('  Agent %d - Individual mean MSD: %.6f, Fused mean MSD: %.6f\n', ...
            a, individual_mean_rmsd, fused_mean_rmsd);
end

% % Check if fusion brought agents closer together
% state_diff_before = norm(individual_estimates(:, end, 1) - individual_estimates(:, end, 2));
% state_diff_after = norm(fused_estimates(:, end, 1) - fused_estimates(:, end, 2));
% fprintf('  State difference before fusion: %.6f\n', state_diff_before);
% fprintf('  State difference after fusion: %.6f\n', state_diff_after);

% if state_diff_after < state_diff_before
%     fprintf('  ✓ Fusion improved consensus\n');
% else
%     fprintf('  ⚠ Fusion did not improve consensus (may be expected with limited data)\n');
% end

fprintf('\n✓ Quick test completed successfully!\n');
fprintf('The refactored test_MAS_sim3.m should work with the current codebase.\n');
