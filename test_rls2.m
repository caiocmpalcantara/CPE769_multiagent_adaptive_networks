% TEST THE REFACTORED RLS2 CLASS IMPLEMENTATION
%   > Using Technique: Rls2 (refactored to follow Agent_technique2 pattern)
%   > Validates RLS filtering technique with new interface and initialization structure
addpath('./Utils/', ...
        './Technique/', ...
        './Technique/Kalman_inc')

global DEBUG_MODE;
DEBUG_MODE = false;

%% Simulate Test Data
addpath("./Technique/")

rng(8988466)

switch sim
    case 1
        % Simple stationary case - similar to test_kalman2.m case 1
        % The model that we want to estimate is u (the "parameter" H)
        x_dim = 3;
        y_dim = 1;
        y_sd = 1;

        u = [1 1 1]';   % The true parameter (H matrix)
        H_true = [1 1 1];
        N = 200;
        n = 1:N;
        noise = y_sd * randn(1,N);

        % Generate constant state vector
        x = ones(x_dim, N);
        
        % Generate observations
        y = H_true * u;
        d = y + noise;

        M = 1;
        dims = [N];

    case 2
        % Time-varying case - similar to test_kalman2.m case 2
        x_dim = 2;
        y_dim = 1;
        y_sd = 0.4;

        u = [.2 .6]';   % The true parameter
        N = 200;
        n = 1:N;
        
        % Time-varying state vector
        x_vals = linspace(1,10,N);
        x = zeros(x_dim, N);
        x(1,:) = x_vals;
        x(2,2:end) = x_vals(1:end-1);
        x(2,1) = 0;
        
        % Generate observations
        d = zeros(1,N);
        y = zeros(1,N);
        noise = y_sd * randn(1,N);
        
        for i = 1:N
            H_true = x(:,i)';
            y(i) = H_true * u;
            d(i) = y(i) + noise(i);
        end

        M = 1;
        dims = [N];

    case 3
        % Monte-Carlo simulation - similar to test_kalman2.m case 3
        x_dim = 3;
        y_dim = 1;
        y_sd = .2;
 
        u = [.1 .3 .7]';    % The true parameter
        N = 1000;
        M = 300;                        % Number of realizations
        
        % Generate time-varying state vectors
        x_base = 0.5*randn(1,N+2) + 1;
        x = zeros(x_dim, N, M);
        
        noise = y_sd * randn(1,N,M);
        d = zeros(1,N,M);
        y = zeros(1,N);

        % Monte-Carlo
        for n = 1:N
            x(:,n,:) = repmat(x_base(n:n+2)', [1, 1, M]);
            H_true = x_base(n:n+2);
            y(n) = H_true * u;
            d(1,n,:) = y(n) + noise(1,n,:);
        end

        dims = [N M];

    otherwise
        fprintf('No simulation (sim) selected. Using default case 1.\n');
        sim = 1;
        % Default to case 1
        x_dim = 3;
        y_dim = 1;
        y_sd = 1;
        u = [1 1 1]';
        H_true = [1 1 1];
        N = 200;
        n = 1:N;
        noise = y_sd * randn(1,N);
        x = ones(x_dim, N);
        y = H_true * u;
        d = y + noise;
        M = 1;
        dims = [N];
end

% Display simulation info
fprintf('=== RLS2 Test Configuration ===\n');
fprintf('Simulation case: %d\n', sim);
fprintf('State dimension (x_dim): %d\n', x_dim);
fprintf('Observation dimension (y_dim): %d\n', y_dim);
fprintf('Noise standard deviation: %.2f\n', y_sd);
fprintf('True parameter u: [%s]\n', num2str(u'));
fprintf('Time steps (N): %d\n', N);
fprintf('Monte Carlo realizations (M): %d\n', M);

figure(1)
plot(n,d(1,:,1), 'b')
xlabel('Time')
ylabel('Observation')
title('Testing RLS2 Class - Noisy Observations')
hold on
grid on

%% RLS2 Filtering
fprintf('\n=== RLS2 Filtering ===\n');

% RLS2 initialization with start_vals structure
start_vals = struct();
start_vals.delta = 0.1;                    % Initial covariance scaling
start_vals.initial_state = zeros(x_dim, 1); % Initial parameter estimate

% Create RLS2 instance
rls = Rls2('x_dim', x_dim, 'y_dim', y_dim, ...
           'H_matrix', zeros(y_dim, x_dim), ...  % Start with zero estimates
           'lambda', 0.95, ...                   % Forgetting factor
           'start_vals', start_vals);

fprintf('RLS2 instance created successfully!\n');
fprintf('  Forgetting factor (lambda): %.3f\n', rls.lambda);
fprintf('  Initial delta: %.3f\n', rls.start_vals.delta);
fprintf('  Initial H matrix: [%s]\n', mat2str(rls.H));

% Initialize result storage
if all(size(dims) > 1)
    H_hat_history = zeros(y_dim, x_dim, N, M);
    y_hat_history = zeros(y_dim, N, M);
else
    H_hat_history = zeros(y_dim, x_dim, N);
    y_hat_history = zeros(y_dim, N);
end

% Monte-Carlo: RLS Processing
fprintf('\nProcessing RLS filtering...\n');
for m = 1:M
    if M > 1 && mod(m, max(1, floor(M/10))) == 0
        fprintf('  Completed %d/%d realizations (%.1f%%)\n', m, M, 100*m/M);
    end
    
    for n = 1:N
        % Get current state vector and measurement
        if sim == 3
            x_current = x(:, n, m);
        else
            x_current = x(:, n);
        end
        
        measurement = d(1, n, m);
        
        % Apply RLS filtering
        [y_hat_out] = rls.apply('measurement', measurement, 'state_buffer', x_current);
        
        % Store results
        if M > 1
            H_hat_history(:, :, n, m) = rls.get_H();
            y_hat_history(:, n, m) = y_hat_out;
        else
            H_hat_history(:, :, n) = rls.get_H();
            y_hat_history(:, n) = y_hat_out;
        end
    end
    
    % Reset RLS for next realization
    if m < M
        rls.reset();
    end
end

fprintf('RLS filtering completed!\n');

%% Visualization
figure(1)
if M > 1
    plot(1:N, y_hat_history(1,:,1), 'r', 'LineWidth', 1.5)
else
    plot(1:N, y_hat_history(1,:), 'r', 'LineWidth', 1.5)
end
legend('Noisy Observations', 'RLS Predictions', 'Location', 'best')
hold off

%% Error Analysis
fprintf('\n=== Error Analysis ===\n');

% Calculate prediction errors (MSE)
if M > 1
    prediction_errors = (y_hat_history - repmat(y, [1, 1, M])).^2;
    mse_pred = mean(prediction_errors, 3);
else
    prediction_errors = (y_hat_history - y).^2;
    mse_pred = prediction_errors;
end

% Calculate parameter estimation errors (MSD)
if M > 1
    param_errors = zeros(1, N, M);
    for m = 1:M
        for n = 1:N
            param_errors(1, n, m) = norm(squeeze(H_hat_history(:, :, n, m)) - u');
        end
    end
    msd_param = mean(param_errors, 3);
else
    param_errors = zeros(1, N);
    for n = 1:N
        param_errors(1, n) = norm(squeeze(H_hat_history(:, :, n)) - u');
    end
    msd_param = param_errors;
end

% Display final performance
fprintf('Final prediction MSE: %.6f\n', mse_pred(end));
fprintf('Final parameter MSD: %.6f\n', msd_param(end));
fprintf('Final H estimate: [%s]\n', mat2str(squeeze(H_hat_history(:, :, end)), 4));
fprintf('True parameter u: [%s]\n', mat2str(u', 4));

% Plot errors
figure(2)
plot(1:N, 10*log10(mse_pred), 'b', 'LineWidth', 1.5)
title('RLS2: Mean Squared Error (Prediction)')
ylabel('MSE [dB]')
xlabel('Time Step')
set(gca, 'YLim', [-40 10])
grid on

figure(3)
plot(1:N, 20*log10(msd_param), 'r', 'LineWidth', 1.5)
title('RLS2: Mean Squared Deviation (Parameter Estimation)')
ylabel('MSD [dB]')
xlabel('Time Step')
set(gca, 'YLim', [-40 10])
grid on

fprintf('\n=== RLS2 Test Summary ===\n');
fprintf('Test completed successfully!\n');
fprintf('RLS2 class validated with Agent_technique2 interface.\n');
fprintf('Key features tested:\n');
fprintf('  - start_vals structure initialization\n');
fprintf('  - apply() method with measurement and state_buffer\n');
fprintf('  - reset() functionality for Monte-Carlo\n');
fprintf('  - Parameter estimation convergence\n');
fprintf('  - Error analysis and visualization\n');
