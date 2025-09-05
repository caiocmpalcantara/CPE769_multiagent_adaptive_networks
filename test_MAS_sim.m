% Initial test Multi-Agent System

clear all;
close all;
clc

addpath("./Technique/")
addpath("./Technique/Kalman_inc/")
addpath("./Agent/")
addpath("./Utils/")

global DEBUG_MODE;
DEBUG_MODE = false;

sim = 6;
fprintf('=== Multi-Agent Social Learning Test (Agent) ===\n');
fprintf('Validating multi-agent social learning with:\n');
fprintf('  - Simulation case %d\n', sim);
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

%% Performance Analysis
fprintf('\n=== Performance Analysis ===\n');

% Calculate errors for each agent
prediction_errors = zeros(N, Na, M);
H_estimation_errors = zeros(N, Na, M);

for a = 1:Na
    % Prediction errors (comparing predicted observations to true observations)
    for m = 1:M
        for t = 1:N
            prediction_errors(t, a, m) = abs(y_hat_history(1, t, a, m) - y(t));
        end
    end

    % H estimation errors (comparing estimated H to true H)
    for m = 1:M
        for t = 1:N
            H_estimation_errors(t, a, m) = norm(H_hat_history(:, :, t, a, m) - u);
        end
    end

    % Report final errors for this agent
    H_estimation_errors_mean = mean(H_estimation_errors, 3);
    fprintf('Agent %d - Final H estimation error: %.6f\n', ...
            a, H_estimation_errors_mean(end, a));
end

% Overall performance metrics
mean_H_error = mean(sqrt(mean(mean(H_estimation_errors, 3).^2, 1)));
mean_pred_error = mean(mean(mean(prediction_errors, 3), 1));

fprintf('\nOverall Performance:\n');
fprintf('  Mean H estimation error (RMS): %.6f\n', mean_H_error);
fprintf('  Mean prediction error: %.6f\n', mean_pred_error);

% Additional metrics
fprintf('\nAdditional Metrics:\n');
fprintf('  True H matrix: [%s]\n', num2str(u));
fprintf('  True observation values: %.6f (constant)\n', y(1));

%% Visualization
fprintf('\nGenerating plots...\n');
m_vis = 1; % Visualization realization
n = 1:N;

% Figure 1: Comprehensive Analysis
figure(1);
clf;

% Subplot 1: Observations and Predictions
subplot(2,2,1);
plot(n, squeeze(d(1, :, 1, m_vis)), 'b-', 'LineWidth', 1);
hold on;
colors = {'r-', 'g-', 'c-', 'm-', 'k-', 'y-'};
for a = 1:min(Na, 6) % Show up to 6 agents
    plot(n, squeeze(y_hat_history(1, :, a, m_vis)), colors{a}, 'LineWidth', 1.5);
end
plot(n, y, 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Observation');
title(sprintf('Observations vs Predictions (realization %d)', m_vis));
legend_entries = {'Noisy Obs (Agent 1)'};
for a = 1:min(Na, 6)
    legend_entries{end+1} = sprintf('Agent %d Pred', a);
end
legend_entries{end+1} = 'True Value';
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 2: H Estimation Convergence
subplot(2,2,2);
for a = 1:min(Na, 4) % Show up to 4 agents for clarity
    plot(n, squeeze(H_hat_history(1, 1, :, a, m_vis)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
plot(n, u(1)*ones(size(n)), 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('H Estimate (Component 1,1)');
title(sprintf('H Estimation Convergence (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend_entries{end+1} = 'True Value';
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 3: H Estimation Errors
subplot(2,2,3);
for a = 1:min(Na, 4)
    plot(n, squeeze(H_estimation_errors(:, a, m_vis)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error');
title(sprintf('H Estimation Errors (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 4: Prediction Errors
subplot(2,2,4);
for a = 1:min(Na, 4)
    plot(n, squeeze(prediction_errors(:, a, m_vis)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 2: Detailed Error Analysis
figure(2);
clf;

% Subplot 1: H Estimation Errors (Linear Scale)
subplot(2,2,1);
for a = 1:min(Na, 4)
    plot(n, squeeze(H_estimation_errors(:, a, m_vis)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error');
title(sprintf('H Estimation Errors - Linear Scale (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 2: H Estimation Errors (dB Scale)
subplot(2,2,2);
for a = 1:min(Na, 4)
    H_errors_db = 20*log10(max(squeeze(H_estimation_errors(:, a, m_vis)), 1e-10));
    plot(n, H_errors_db, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error [dB]');
title(sprintf('H Estimation Errors - dB Scale (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 3: Prediction Errors (Linear Scale)
subplot(2,2,3);
for a = 1:min(Na, 4)
    plot(n, squeeze(prediction_errors(:, a, m_vis)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors - Linear Scale (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Subplot 4: Prediction Errors (dB Scale)
subplot(2,2,4);
for a = 1:min(Na, 4)
    pred_errors_db = 10*log10(max(squeeze(prediction_errors(:, a, m_vis)).^2, 1e-10));
    plot(n, pred_errors_db, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error [dB]');
title(sprintf('Prediction Errors - dB Scale (realization %d)', m_vis));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 4: Final Performance Comparison
figure(4);
clf;

% Figure 4: Subplot 1: Final H Estimation Errors
subplot(1,2,1);
agents_idx = 1:Na;
final_H_errors = squeeze(H_estimation_errors(end, :, m_vis));
bar(agents_idx, final_H_errors, 'FaceColor', 'b', 'FaceAlpha', 0.7);
xlabel('Agent');
ylabel('Final H Estimation Error');
title(sprintf('Final H Estimation Performance (realization %d)', m_vis));
grid on;

% Figure 4: Subplot 2: Mean H Estimation Errors (RMS)
subplot(1,2,2);
mean_H_errors = sqrt(mean(squeeze(H_estimation_errors(:, :, m_vis)).^2, 1));
bar(agents_idx, mean_H_errors, 'FaceColor', 'r', 'FaceAlpha', 0.7);
xlabel('Agent');
ylabel('Mean H Estimation Error (RMS)');
title(sprintf('Mean H Estimation Performance (realization %d)', m_vis));
grid on;

fprintf('\n=== Test Summary (realization %d) ===\n', m_vis);
fprintf('Test completed successfully!\n');
fprintf('\nKey findings:\n');

% Check convergence
convergence_threshold = 0.1;
converged_agents = sum(squeeze(H_estimation_errors(end, :, m_vis)) < convergence_threshold);
fprintf('  - Agents converged (H error < %.1f) (realization %d): %d/%d\n', ...
        convergence_threshold, m_vis, converged_agents, Na);

% Final performance summary
final_H_errors = squeeze(H_estimation_errors(end, :, m_vis));
fprintf('  - Best performing agent: Agent %d (error: %.6f)\n', ...
        find(final_H_errors == min(final_H_errors), 1), min(final_H_errors));
fprintf('  - Worst performing agent: Agent %d (error: %.6f)\n', ...
        find(final_H_errors == max(final_H_errors), 1), max(final_H_errors));

%% Visualization Monte-Carlo
fprintf('\nGenerating Monte-Carlo averaged plots...\n');

% Figure 3: Detailed Error Analysis
figure(3);
clf;

% Figure 3: Subplot 1: H Estimation Errors (Linear Scale)
subplot(2,2,1);
for a = 1:min(Na, 4)
    plot(n, squeeze(mean(H_estimation_errors(:, a, :), 3)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error');
title(sprintf('H Estimation Errors - Linear Scale (Monte-Carlo)'));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 3: Subplot 2: H Estimation Errors (dB Scale)
subplot(2,2,2);
for a = 1:min(Na, 4)
    H_errors_db = 20*log10(max(squeeze(mean(H_estimation_errors(:, a, :), 3)), 1e-10));
    plot(n, H_errors_db, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error (RMS) [dB]');
title(sprintf('H Estimation Errors - dB Scale (Monte-Carlo)'));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 3: Subplot 3: Prediction Errors (Linear Scale)
subplot(2,2,3);
for a = 1:min(Na, 4)
    plot(n, squeeze(mean(prediction_errors(:, a, :), 3)), colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error');
title(sprintf('Prediction Errors - Linear Scale (Monte-Carlo)'));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 3: Subplot 4: Prediction Errors (dB Scale)
subplot(2,2,4);
for a = 1:min(Na, 4)
    pred_errors_db = 10*log10(max(squeeze(mean(prediction_errors(:, a, :), 3)).^2, 1e-10));
    plot(n, pred_errors_db, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error [dB]');
title(sprintf('Prediction Errors - dB Scale (Monte-Carlo)'));
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;


% Figure 5: Monte-Carlo Analysis
figure(5);
clf;

% Figure 5: Subplot 1: H Estimation Convergence (Monte-Carlo)
subplot(2,2,1);
for a = 1:min(Na, 4)
    H_conv_mc = squeeze(mean(H_hat_history(1, 1, :, a, :), 5));
    plot(n, H_conv_mc, colors{a}, 'LineWidth', 1.5);
    hold on;
end
plot(n, u(1)*ones(size(n)), 'k--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('H Estimate (Component 1,1)');
title('H Estimation Convergence - Monte-Carlo');
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend_entries{end+1} = 'True Value';
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 5: Subplot 2: H Estimation Errors (Monte-Carlo)
subplot(2,2,2);
for a = 1:min(Na, 4)
    H_errors_mc = mean(H_estimation_errors(:, a, :), 3);
    plot(n, H_errors_mc, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error');
title('H Estimation Errors - Monte-Carlo');
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 5: Subplot 3: H Estimation Errors in dB (Monte-Carlo)
subplot(2,2,3);
for a = 1:min(Na, 4)
    H_errors_mc = mean(H_estimation_errors(:, a, :), 3);
    H_errors_db = 20*log10(max(H_errors_mc, 1e-10));
    plot(n, H_errors_db, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('H Estimation Error [dB]');
title('H Estimation Errors [dB] - Monte-Carlo');
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 5: Subplot 4: Prediction Errors (Monte-Carlo)
subplot(2,2,4);
for a = 1:min(Na, 4)
    pred_errors_mc = mean(prediction_errors(:, a, :), 3);
    plot(n, pred_errors_mc, colors{a}, 'LineWidth', 1.5);
    hold on;
end
xlabel('Time Step');
ylabel('Prediction Error');
title('Prediction Errors - Monte-Carlo');
legend_entries = {};
for a = 1:min(Na, 4)
    legend_entries{end+1} = sprintf('Agent %d', a);
end
legend(legend_entries, 'Location', 'best');
grid on;

% Figure 6: Monte-Carlo Performance Comparison
figure(6);
clf;

% Figure 6: Subplot 1: Final H Estimation Errors (Monte-Carlo)
subplot(1,2,1);
agents_idx = 1:Na;
final_H_errors_mc = mean(H_estimation_errors(end, :, :), 3);
bar(agents_idx, final_H_errors_mc, 'FaceColor', 'b', 'FaceAlpha', 0.7);
xlabel('Agent');
ylabel('Final H Estimation Error');
title('Final H Estimation Performance - Monte-Carlo');
grid on;

% Figure 6: Subplot 2: Mean H Estimation Errors (RMS, Monte-Carlo)
subplot(1,2,2);
mean_H_errors_mc = sqrt(mean(mean(H_estimation_errors, 3).^2, 1));
bar(agents_idx, mean_H_errors_mc, 'FaceColor', 'r', 'FaceAlpha', 0.7);
xlabel('Agent');
ylabel('Mean H Estimation Error (RMS)');
title('Mean H Estimation Performance - Monte-Carlo');
grid on;

fprintf('\n=== Test Summary (Monte-Carlo) ===\n');
fprintf('Test completed successfully!\n');
fprintf('\nKey findings:\n');

% Overall Monte-Carlo performance
mean_H_error_mc = mean(sqrt(mean(mean(H_estimation_errors, 3).^2, 1)));
mean_pred_error_mc = mean(mean(mean(prediction_errors, 3), 1));

fprintf('  - Overall mean H estimation error (RMS): %.6f\n', mean_H_error_mc);
fprintf('  - Overall mean prediction error: %.6f\n', mean_pred_error_mc);

% Convergence analysis (Monte-Carlo)
convergence_threshold = 0.1;
final_H_errors_mc = mean(H_estimation_errors(end, :, :), 3);
converged_agents_mc = sum(final_H_errors_mc < convergence_threshold);
fprintf('  - Agents converged (H error < %.1f) (Monte-Carlo): %d/%d\n', ...
        convergence_threshold, converged_agents_mc, Na);

% Best and worst performing agents (Monte-Carlo)
[min_error, best_agent] = min(final_H_errors_mc);
[max_error, worst_agent] = max(final_H_errors_mc);
fprintf('  - Best performing agent (Monte-Carlo): Agent %d (error: %.6f)\n', ...
        best_agent, min_error);
fprintf('  - Worst performing agent (Monte-Carlo): Agent %d (error: %.6f)\n', ...
        worst_agent, max_error);

% Performance improvement analysis
if Na > 1
    error_std = std(final_H_errors_mc);
    error_range = max_error - min_error;
    fprintf('  - Performance variation: std=%.6f, range=%.6f\n', error_std, error_range);
end

fprintf('\nSimulation validated multi-agent parameter estimation with:\n');
fprintf('  - True H matrix: [%s]\n', num2str(u));
fprintf('  - %d agents with different cooperation strategies\n', Na);
fprintf('  - %d time steps, %d Monte Carlo realizations\n', N, M);

% Compatibility note for comparison with test_MAS_sim3.m
fprintf('\n=== Compatibility with test_MAS_sim3.m ===\n');
fprintf('This analysis provides comparable metrics to test_MAS_sim3.m:\n');
fprintf('  - Parameter estimation errors (H matrix vs state vector)\n');
fprintf('  - Prediction performance analysis\n');
fprintf('  - Monte-Carlo statistical validation\n');
fprintf('  - Agent performance comparison\n');
fprintf('Use these results to compare different multi-agent learning approaches.\n');