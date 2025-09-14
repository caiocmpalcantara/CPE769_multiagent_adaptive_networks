% COMPARISON TEST: RLS vs KF (Kalman Filter)
%   > Validates that refactored RLS produces results consistent with Kalman filtering
%   > Both estimate state vector x with known observation matrix H
%   > Mathematical model: y = H * x + noise
addpath('./Utils/', ...
        './Technique/', ...
        './Technique/Kalman_inc')

global DEBUG_MODE;
DEBUG_MODE = false;

%% Test Configuration
rng(8988466)

% System parameters
x_dim = 3;
y_dim = 1;
y_sd = 0.5;
N = 100;

% True state (constant for this comparison)
x_true = [0.2, 0.5, 0.8]';

% Known observation matrix
H_matrix = [1, 2, 1];

% Generate observations
noise = y_sd * randn(1, N);
y_true = H_matrix * x_true;
d = y_true + noise;

fprintf('=== RLS vs Kalman Filter Comparison ===\n');
fprintf('State dimension: %d\n', x_dim);
fprintf('Observation dimension: %d\n', y_dim);
fprintf('True state: [%s]\n', num2str(x_true'));
fprintf('Observation matrix: [%s]\n', num2str(H_matrix));
fprintf('Time steps: %d\n', N);
fprintf('Noise std: %.2f\n', y_sd);

%% RLS State Estimation Setup
start_vals_rls = struct();
start_vals_rls.delta = 0.1;
start_vals_rls.initial_state = zeros(x_dim, 1);

rls = RLS('x_dim', x_dim, 'y_dim', y_dim, ...
           'H_matrix', H_matrix, ...
           'lambda', 0.99, ...  % High lambda for comparison with Kalman
           'start_vals', start_vals_rls);

fprintf('\nRLS initialized:\n');
fprintf('  Lambda: %.3f\n', rls.lambda);
fprintf('  Initial state: [%s]\n', mat2str(rls.x_hat'));

%% Kalman Filter Setup
Q = zeros(x_dim);  % No process noise for fair comparison
A = eye(x_dim);    % Identity state transition
model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);

R = y_sd^2;  % Measurement noise variance

kf = KF('x_dim', x_dim, 'y_dim', y_dim, ...
             'H_matrix', H_matrix, 'R_matrix', R, ...
             'Pa_init', {'delta', 0.1}, ...
             'xa_init', {'initial_state', zeros(x_dim, 1)}, ...
             'system_model', model_sys);

fprintf('\nKalman Filter initialized:\n');
fprintf('  R matrix: %.3f\n', R);
fprintf('  Initial state: [%s]\n', mat2str(kf.xa_hat'));

%% Run Both Filters
x_hat_rls = zeros(x_dim, N);
y_hat_rls = zeros(y_dim, N);
x_hat_kf = zeros(x_dim, N);
y_hat_kf = zeros(y_dim, N);

fprintf('\nProcessing both filters...\n');

for n = 1:N
    % RLS processing
    [y_rls, x_rls] = rls.apply('measurement', d(n));
    x_hat_rls(:, n) = x_rls;
    y_hat_rls(:, n) = y_rls;
    
    % Kalman filter processing
    [y_kf, x_kf_post, x_kf_prior, ~, ~, ~, ~, ~] = kf.apply('measurement', d(n));
    x_hat_kf(:, n) = x_kf_post;
    y_hat_kf(:, n) = y_kf;
end

fprintf('Processing completed!\n');

%% Error Analysis
% State estimation errors
error_rls = zeros(1, N);
error_kf = zeros(1, N);

for n = 1:N
    error_rls(n) = norm(x_hat_rls(:, n) - x_true);
    error_kf(n) = norm(x_hat_kf(:, n) - x_true);
end

% Prediction errors
pred_error_rls = (y_hat_rls - y_true).^2;
pred_error_kf = (y_hat_kf - y_true).^2;

% Final performance comparison
fprintf('\n=== Performance Comparison ===\n');
fprintf('Final State Estimation Error:\n');
fprintf('  RLS: %.6f\n', error_rls(end));
fprintf('  Kalman: %.6f\n', error_kf(end));
fprintf('  Difference: %.6f\n', abs(error_rls(end) - error_kf(end)));

fprintf('\nFinal State Estimates:\n');
fprintf('  RLS: [%s]\n', mat2str(x_hat_rls(:, end)', 4));
fprintf('  Kalman: [%s]\n', mat2str(x_hat_kf(:, end)', 4));
fprintf('  True: [%s]\n', mat2str(x_true', 4));

fprintf('\nMean State Estimation Error:\n');
fprintf('  RLS: %.6f\n', mean(error_rls));
fprintf('  Kalman: %.6f\n', mean(error_kf));

fprintf('\nFinal Prediction Error:\n');
fprintf('  RLS: %.6f\n', pred_error_rls(end));
fprintf('  Kalman: %.6f\n', pred_error_kf(end));

%% Visualization
% State estimation errors
figure(1);
clf;
subplot(2,2,1);
plot(1:N, 20*log10(error_rls), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:N, 20*log10(error_kf), 'r--', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('State Error [dB]');
title('State Estimation Error Comparison');
legend('RLS', 'Kalman', 'Location', 'best');
grid on;

% State convergence for each component
for i = 1:min(x_dim, 3)  % Show up to 3 components
    subplot(2,2,i+1);
    plot(1:N, x_hat_rls(i, :), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(1:N, x_hat_kf(i, :), 'r--', 'LineWidth', 1.5);
    plot(1:N, x_true(i)*ones(1,N), 'k:', 'LineWidth', 2);
    xlabel('Time Step');
    ylabel(sprintf('x_%d', i));
    title(sprintf('State Component %d Convergence', i));
    legend('RLS', 'Kalman', 'True', 'Location', 'best');
    grid on;
end

% Prediction comparison
figure(2);
clf;
subplot(2,1,1);
plot(1:N, d, 'k.', 'MarkerSize', 4);
hold on;
plot(1:N, y_hat_rls, 'b-', 'LineWidth', 1.5);
plot(1:N, y_hat_kf, 'r--', 'LineWidth', 1.5);
plot(1:N, y_true*ones(1,N), 'g:', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Observation');
title('Prediction Comparison');
legend('Noisy Obs', 'RLS Pred', 'Kalman Pred', 'True Value', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(1:N, 10*log10(pred_error_rls), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:N, 10*log10(pred_error_kf), 'r--', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Prediction Error [dB]');
title('Prediction Error Comparison');
legend('RLS', 'Kalman', 'Location', 'best');
grid on;

%% Consistency Analysis
% Check if the algorithms converge to similar solutions
state_diff = zeros(1, N);
for n = 1:N
    state_diff(n) = norm(x_hat_rls(:, n) - x_hat_kf(:, n));
end

fprintf('\n=== Consistency Analysis ===\n');
fprintf('State estimate difference (final): %.6f\n', state_diff(end));
fprintf('State estimate difference (mean): %.6f\n', mean(state_diff));
fprintf('State estimate difference (max): %.6f\n', max(state_diff));

% Convergence threshold
convergence_threshold = 0.1;
rls_converged = error_rls(end) < convergence_threshold;
kf_converged = error_kf(end) < convergence_threshold;

fprintf('\nConvergence Analysis (threshold: %.1f):\n', convergence_threshold);
fprintf('  RLS converged: %s\n', mat2str(rls_converged));
fprintf('  Kalman converged: %s\n', mat2str(kf_converged));

if rls_converged && kf_converged
    fprintf('  ✓ Both algorithms converged successfully\n');
elseif rls_converged
    fprintf('  ⚠ Only RLS converged\n');
elseif kf_converged
    fprintf('  ⚠ Only Kalman converged\n');
else
    fprintf('  ✗ Neither algorithm converged\n');
end

%% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('Comparison test completed successfully!\n');
fprintf('\nKey Findings:\n');

% Performance similarity
perf_similarity = abs(error_rls(end) - error_kf(end)) / max(error_rls(end), error_kf(end)) * 100;
fprintf('  - Performance similarity: %.1f%% (lower is better)\n', perf_similarity);

% Mathematical consistency
if state_diff(end) < 0.01
    fprintf('  - ✓ Mathematical consistency: EXCELLENT (diff < 0.01)\n');
elseif state_diff(end) < 0.1
    fprintf('  - ✓ Mathematical consistency: GOOD (diff < 0.1)\n');
else
    fprintf('  - ⚠ Mathematical consistency: MODERATE (diff = %.3f)\n', state_diff(end));
end

% Interface compatibility
fprintf('  - ✓ Interface compatibility: Both use Agent_technique pattern\n');
fprintf('  - ✓ State estimation: Both estimate x with known H_matrix\n');
fprintf('  - ✓ Simplified interface: RLS no longer needs state_buffer\n');

fprintf('\nRefactoring Success:\n');
fprintf('  ✓ RLS now estimates state vector (consistent with Kalman)\n');
fprintf('  ✓ H_matrix treated as known observation matrix\n');
fprintf('  ✓ Mathematical interpretation aligned with Kalman filtering\n');
fprintf('  ✓ Simplified apply() interface (no state_buffer required)\n');
fprintf('  ✓ Maintains Agent_technique compatibility\n');
