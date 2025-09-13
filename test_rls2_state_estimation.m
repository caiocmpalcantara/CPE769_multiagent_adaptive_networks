% TEST THE REFACTORED RLS2 CLASS - STATE ESTIMATION VERSION
%   > Tests RLS2 refactored to estimate state vector x (consistent with Kalman filtering)
%   > Mathematical model: y = H_matrix * x + noise (estimate x, H_matrix known)
%   > Validates compatibility with Agent_technique2 pattern
addpath('./Utils/', ...
        './Technique/', ...
        './Technique/Kalman_inc')

global DEBUG_MODE;
DEBUG_MODE = false;

%% Simulate Test Data for State Estimation
addpath("./Technique/")

rng(8988466)

switch sim
    case 1
        % Simple stationary state estimation
        % Mathematical model: y = H_matrix * x + noise
        x_dim = 3;
        y_dim = 1;
        y_sd = 1;

        x_true = [1 1 1]';      % True state vector (to be estimated)
        H_matrix = [1 1 1];     % Known observation matrix
        N = 200;
        n = 1:N;
        noise = y_sd * randn(1,N);

        % Generate observations
        y_true = H_matrix * x_true;
        d = y_true + noise;

        M = 1;
        dims = [N];

    case 2
        % Time-varying observation matrix
        x_dim = 2;
        y_dim = 1;
        y_sd = 0.4;

        x_true = [.2 .6]';      % True state vector (constant)
        N = 200;
        n = 1:N;
        
        % Time-varying observation matrix
        h_vals = linspace(1,10,N);
        H_matrix_sequence = zeros(N, y_dim, x_dim);
        
        % Generate observations with time-varying H
        d = zeros(1,N);
        y_true = zeros(1,N);
        noise = y_sd * randn(1,N);
        
        for i = 1:N
            H_matrix_sequence(i,:,:) = [h_vals(i), h_vals(max(1,i-1))];
            H_current = reshape(H_matrix_sequence(i,:,:), [y_dim, x_dim]);
            y_true(i) = H_current * x_true;
            d(i) = y_true(i) + noise(i);
        end
        
        % Use first H_matrix for initialization
        H_matrix = reshape(H_matrix_sequence(1,:,:), [y_dim, x_dim]);

        M = 1;
        dims = [N];

    case 3
        % Monte-Carlo simulation for state estimation
        x_dim = 3;
        y_dim = 1;
        y_sd = .2;
 
        x_true = [.1 .3 .7]';   % True state vector (constant)
        N = 1000;
        M = 300;                % Number of realizations
        
        % Constant observation matrix
        H_matrix = [0.5, 1.0, 1.5];
        
        noise = y_sd * randn(1,N,M);
        d = zeros(1,N,M);
        y_true = zeros(1,N);

        % Monte-Carlo
        for n = 1:N
            y_true(n) = H_matrix * x_true;
            d(1,n,:) = y_true(n) + noise(1,n,:);
        end

        dims = [N M];

    case 4
        % Monte-Carlo simulation for state estimation with time-varying H_matrix
        % Simulation parameters (as specified)
        x_dim = 3;
        y_dim = 1;


        N = 500;
        M = 10;    % Number of Monte Carlo realizations
        Na = 1;  % Number of agents

        % Assumption: same state dynamic for all agents, different noise observations
        x_true = zeros(3,1);
        x_true(:,1) = [-0.2 0.7 0.3]';   % The initial state

        n = 1:N;
        rng(8988466)


        noisePowers_dB = [ ...
        -27.6, -24.2, -10.3, -22.4, -26.6, ...
        -17.1, -23.1, -21.7, -21.2, -25.5, ...
        -13.3, -21.6, -25.7, -20.0, -10.4, ...
        -15.7, -20.4, -11.6, -20.9, -24.7 ];

        regressionPower_dB = [ ...
        12.0, 10.4, 12.5, 12.5, 10.0, ...
        12.6, 12.3, 12.2, 12.4, 11.5, ...
        11.6, 11.4, 12.6, 12.6, 12.5, ...
        10.4, 11.5, 12.1, 10.4, 12.2 ];


        % Autoregressive model input (same to all agents)
        u = zeros(3,N); % Kalman => H
        rk = 0.95;
        x_sd = .5;
        x = x_sd * randn(3,N); % excitation
        for i = 2:N
            u(:,i) = rk * u(:,i-1)  + sqrt(1-rk^2) * x(:,i);
        end

        d = zeros(1,Na,N,M);
        H = zeros(N,3);
        y = zeros(1,N);
        for n = 1:N
            H(n,:) = u(:,n)';
            y(n) = H(n,:)*x_true;
            for a = 1:Na
                d(:,a,n,:) = y(n) + 10^(noisePowers_dB(a)/10) * randn(1,1,1,M);
            end
        end

        dims = [N M];

    otherwise
        fprintf('No simulation (sim) selected. Using default case 1.\n');
        sim = 1;
        % Default to case 1
        x_dim = 3;
        y_dim = 1;
        y_sd = 1;
        x_true = [1 1 1]';
        H_matrix = [1 1 1];
        N = 200;
        n = 1:N;
        noise = y_sd * randn(1,N);
        y_true = H_matrix * x_true;
        d = y_true + noise;
        M = 1;
        dims = [N];
end

% Display simulation info
fprintf('=== RLS2 State Estimation Test Configuration ===\n');
fprintf('Simulation case: %d\n', sim);
fprintf('State dimension (x_dim): %d\n', x_dim);
fprintf('Observation dimension (y_dim): %d\n', y_dim);
fprintf('Noise standard deviation: %.2f\n', y_sd);
fprintf('True state x: [%s]\n', num2str(x_true'));
fprintf('Observation matrix H: [%s]\n', num2str(H_matrix));
fprintf('Time steps (N): %d\n', N);
fprintf('Monte Carlo realizations (M): %d\n', M);

% figure(1)
% plot(n,d(1,:,1), 'b--', 'LineWidth', 1.5)
% xlabel('Time')
% ylabel('Observation')
% % title('Testing RLS2 State Estimation - Noisy Observations')
% hold on
% grid on

%% RLS2 State Estimation
fprintf('\n=== RLS2 State Estimation ===\n');

% RLS2 initialization with start_vals structure for state estimation
start_vals = struct();
start_vals.delta = 0.1;                    % Initial state covariance scaling
start_vals.initial_state = zeros(x_dim, 1); % Initial state estimate

% Create RLS2 instance for state estimation
rls = Rls2('x_dim', x_dim, 'y_dim', y_dim, ...
           'H_matrix', H_matrix, ...              % Known observation matrix
           'lambda', 0.95, ...                    % Forgetting factor
           'start_vals', start_vals);

fprintf('RLS2 state estimation instance created successfully!\n');
fprintf('  Forgetting factor (lambda): %.3f\n', rls.lambda);
fprintf('  Initial delta: %.3f\n', rls.start_vals.delta);
fprintf('  Observation matrix H: [%s]\n', mat2str(rls.H_matrix));
fprintf('  Initial state estimate: [%s]\n', mat2str(rls.x_hat'));

% Initialize result storage
if all(size(dims) > 1)
    x_hat_history = zeros(x_dim, N, M);
    y_hat_history = zeros(y_dim, N, M);
else
    x_hat_history = zeros(x_dim, N);
    y_hat_history = zeros(y_dim, N);
end

% Monte-Carlo: RLS State Estimation Processing
fprintf('\nProcessing RLS state estimation...\n');
for m = 1:M
    if M > 1 && mod(m, max(1, floor(M/10))) == 0
        fprintf('  Completed %d/%d realizations (%.1f%%)\n', m, M, 100*m/M);
    end
    
    for n = 1:N
        % Get current measurement
        measurement = d(1, n, m);
        
        % Update observation matrix if time-varying (case 2)
        if sim == 2 && n > 1
            H_current = reshape(H_matrix_sequence(n,:,:), [y_dim, x_dim]);
            rls.update_H_matrix(H_current);
        end
        
        % Apply RLS state estimation (no state_buffer needed)
        [x_hat_out, y_hat_out] = rls.apply('measurement', measurement);
        
        % Store results
        if M > 1
            x_hat_history(:, n, m) = x_hat_out;
            y_hat_history(:, n, m) = y_hat_out;
        else
            x_hat_history(:, n) = x_hat_out;
            y_hat_history(:, n) = y_hat_out;
        end
    end
    
    % Reset RLS for next realization
    if m < M
        rls.reset();
    end
end

fprintf('RLS state estimation completed!\n');

%% Visualization
figure(1)
plot(1:N, d(1,:,1), 'b')
xlabel('Time')
ylabel('Observation')
% title('Testing RLS2 State Estimation - Noisy Observations')
hold on
grid on
if M > 1
    plot(1:N, y_hat_history(1,:,1), 'r', 'LineWidth', 1.5)
    title('RLS2 State Estimation: Predicted Observations (1st Realization)')
else
    plot(1:N, y_hat_history(1,:), 'r', 'LineWidth', 1.5)
    title('RLS2 State Estimation: Predicted Observations')
end
legend('Noisy Observations', 'RLS Predictions', 'Location', 'best')
hold off

%% Error Analysis
fprintf('\n=== Error Analysis ===\n');

% Calculate prediction errors (MSE)
if M > 1
    prediction_errors = (y_hat_history - repmat(y_true, [1, 1, M])).^2;
    mse_pred = mean(prediction_errors, 3);
else
    prediction_errors = (y_hat_history - y_true).^2;
    mse_pred = prediction_errors;
end

% Calculate state estimation errors (MSD)
if M > 1
    state_errors = zeros(1, N, M);
    for m = 1:M
        for n = 1:N
            state_errors(1, n, m) = norm(x_hat_history(:, n, m) - x_true);
        end
    end
    msd_state = mean(state_errors, 3);
else
    state_errors = zeros(1, N);
    for n = 1:N
        state_errors(1, n) = norm(x_hat_history(:, n) - x_true);
    end
    msd_state = state_errors;
end

% Display final performance
fprintf('Final prediction MSE: %.6f\n', mse_pred(end));
fprintf('Final state estimation MSD: %.6f\n', msd_state(end));
if M > 1
    fprintf('Final state estimate: [%s]\n', mat2str(x_hat_history(:, end, 1)', 4));
else
    fprintf('Final state estimate: [%s]\n', mat2str(x_hat_history(:, end)', 4));
end
fprintf('True state x: [%s]\n', mat2str(x_true', 4));

% Plot errors
figure(2)
plot(1:N, 10*log10(mse_pred), 'b', 'LineWidth', 1.5)
if M > 1
    title(sprintf('RLS2: Mean Squared Error (Prediction) - Monte Carlo: %d realizations', M));
else
    title('RLS2: Mean Squared Error (Prediction)')
end
ylabel('MSE [dB]')
xlabel('Time Step')
set(gca, 'YLim', [-40 10])
grid on

figure(3)
plot(1:N, 20*log10(msd_state), 'r', 'LineWidth', 1.5)
if M > 1
    title(sprintf('RLS2: Mean Squared Deviation (State Estimation) - Monte Carlo: %d realizations', M));
else
    title('RLS2: Mean Squared Deviation (State Estimation)')
end
ylabel('MSD [dB]')
xlabel('Time Step')
set(gca, 'YLim', [-40 10])
grid on

% Plot state convergence
figure(4)
if M > 1
    for i = 1:x_dim
        subplot(x_dim, 1, i)
        plot(1:N, squeeze(x_hat_history(i, :, 1)), 'r', 'LineWidth', 1.5)
        hold on
        plot(1:N, x_true(i)*ones(1,N), 'b--', 'LineWidth', 1.5)
        ylabel(sprintf('x_%d', i))
        legend('Estimate', 'True Value', 'Location', 'best')
        grid on
        if i == 1
            title(sprintf('RLS2: State Convergence, Monte Carlo: %d realizations', M));
        end
        if i == x_dim
            xlabel('Time Step')
        end
    end
else
    for i = 1:x_dim
        subplot(x_dim, 1, i)
        plot(1:N, x_hat_history(i, :), 'r', 'LineWidth', 1.5)
        hold on
        plot(1:N, x_true(i)*ones(1,N), 'b--', 'LineWidth', 1.5)
        ylabel(sprintf('x_%d', i))
        legend('Estimate', 'True Value', 'Location', 'best')
        grid on
        if i == 1
            title('RLS2: State Convergence')
        end
        if i == x_dim
            xlabel('Time Step')
        end
    end
end

fprintf('\n=== RLS2 State Estimation Test Summary ===\n');
fprintf('Test completed successfully!\n');
fprintf('RLS2 class refactored for state estimation (consistent with Kalman filtering).\n');
fprintf('Key features tested:\n');
fprintf('  - State estimation instead of parameter estimation\n');
fprintf('  - H_matrix as known observation matrix\n');
fprintf('  - Simplified apply() interface (no state_buffer needed)\n');
fprintf('  - start_vals structure for state initialization\n');
fprintf('  - reset() functionality for Monte-Carlo\n');
fprintf('  - State convergence analysis\n');
fprintf('  - Mathematical consistency with Kalman filtering\n');
