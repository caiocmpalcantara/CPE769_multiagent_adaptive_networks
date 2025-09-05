% TEST THE BASIC STRUCTURE OF KALMAN CLASS SIMULATION
%   > Using Utils: GeoPoint, Displacement, Speed, Noise Model
%   > Using Technique: Kalman, KF
%   > Using Technique/Kalman_inc: State Model
addpath('./Utils/', ...
        './Technique/', ...
        './Technique/Kalman_inc')

global DEBUG_MODE;
DEBUG_MODE = false;
%% Simulate a Trajectory
addpath("./Technique/")

rng(8988466)

switch sim
    case 1
        % The model that we want is u (the "state" x)
        % To simulate in order to compare with RLS, LMS
        x_dim = 3;
        y_dim = 1;
        y_sd = 1;

        u = [1 1 1]';   % The "state"
        H = [1 1 1];
        N = 200;
        n = 1:N;
        noise = y_sd * randn(1,N);

        y = H*u;
        
        d = y + noise;

        M = 1;
        dims = [N];

    case 2
        % The model that we want is u (the "state" x)
        % To simulate in order to compare with RLS, LMS
        x_dim = 2;
        y_dim = 1;
        y_sd = 0.4;

        u = [.2 .6]';   % The "state"
        N = 200;
        n = 1:N;
        % x = linspace(3,3,N);
        x = linspace(1,10,N);
        % x = sin(n/10*pi);
        d = zeros(1,N);
        y = zeros(1,N);
        noise = y_sd * randn(1,N);
        H = zeros(N, x_dim);
        H(1,:) = [x(1) 0];
        % y(1) = u*[x(1) 0]';
        
        % Simulate
        y(1) = H(1,:)*u;
        d(1) = y(1) + noise(1);
        for i = 1:N-1
            H(i,:) = [x(i+1) x(i)];
            % y(i+1) = u*[x(i+1) x(i)]';
            y(i+1) = H(i,:)*u;
            d(i+1) = y(i+1) + noise(i+1);
        end

        M = 1;
        dims = [N];

    case 3
        % Simulation with Monte-Carlo
        % The model that we want is u (the "state" x)
        x_dim = 3;
        y_dim = 1;
        y_sd = .2;
 
        u = [.1 .3 .7]';    % The "state"
        N = 1000;
        M = 300;                        % Number of realizations
        x = 0.5*randn(1,N+2) + 1;
        noise = y_sd * randn(1,N,M);
        d = zeros(1,N,M);
        y = zeros(1,N);

        % Monte-Carlo
        for n = 1:N
            H(n,:) = x(n:n+2);
            % y(n) = u*x(n:n+2)';
            y(n) = H*u;
            d(1,n,:) = y(n) + noise(1,n,:);
        end

        dims = [N M];

    otherwise
        fprintf('No simulation (sim) selected.')
end


% d(2:end) = x(2:end);

figure(1)
plot(n,d(1,:,1), 'b')
xlabel('Time')
ylabel('Observation')
title('Testing Kalman2 Class')
hold on
grid on
% plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')

%% Kalman Filtering
% Kalman Init

Q = zeros(x_dim);
A = eye(x_dim);
model_sys = Linear_State('dim', x_dim, 'Q_matrix', Q, 'A_matrix', A);
if length(H(:,1)) > 1
    H_matrix = H(1,:);
else
    H_matrix = H;
end
R = y_sd;

kf = KF_diff('H_matrix', H_matrix, 'R_matrix', R, 'Pa_init', {'delta', 0.1}, ...
            'xa_init', {'initial_state', zeros(x_dim, 1)}, 'system_model', model_sys);


%FIXME: Arrumar daqui pra baixo
% range_sd = 15;  %[m]
% bearing_sd = 2; %[deg]


kf_xp = zeros([x_dim, dims]);
kf_xa = zeros([x_dim, dims]);
if all(size(dims)>1)
    y_hat = zeros(dims);
else
    y_hat = zeros(1,dims);
end


% Monte-Carlo: Kalman Processing (Update new measurement)
for m = 1:M
    for n = 1:N
        [kf_xp(:,n,m), kf_xa(:,n,m), Pp, Pa, a, S, K, y_hat(:,n,m)] = kf.apply('measurement', d(1,n,m));
    end
end


% %% Monte-Carlo sums
% % Position-Dependent variance
% var_rx = range_sd^2*ones(1,N);
% var_ry = range_sd^2*ones(1,N);
% var_vx = (2*var_rx)./(seconds(timestamp(2)-timestamp(1)))^2;
% var_vy = (2*var_ry)./(seconds(timestamp(2)-timestamp(1)))^2;

% % MSE Noisy obs 
% % reshape([pdnkf_v.get_cart(2)], dims)
% obs_mse_rx = mean((d.get_cart(1)' * ones(1,M) - reshape([ship_measure.get_cart(1)], dims)).^2, 2)./var_rx';
% obs_mse_ry = mean((d.get_cart(2)' * ones(1,M) - reshape([ship_measure.get_cart(2)], dims)).^2, 2)./var_ry';
% vx = [ones(1,M); reshape([ship_measure(2:end,:).get_cart(1) - ship_measure(1:end-1,:).get_cart(1)], [N-1,M])/T];
% obs_mse_vx = mean((ones(N,M) * v0.get_cart(1)- vx).^2, 2)./var_vx';
% vy = [ones(1,M); reshape([ship_measure(2:end,:).get_cart(2) - ship_measure(1:end-1,:).get_cart(2)], [N-1,M])/T];
% obs_mse_vy = mean((ones(N,M) * v0.get_cart(2)- vy).^2, 2)./var_vy';

% % MSE KF
% kf_mse_rx = mean((d.get_cart(1)' * ones(1,M) - reshape([kf_r.get_cart(1)], dims)).^2, 2)./var_rx';
% kf_mse_ry = mean((d.get_cart(2)' * ones(1,M) - reshape([kf_r.get_cart(2)], dims)).^2, 2)./var_ry';
% kf_mse_vx = mean((ones(N,M) * v0.get_cart(1) - reshape([kf_v.get_cart(1)], dims)).^2, 2)./var_vx';
% kf_mse_vy = mean((ones(N,M) * v0.get_cart(2) - reshape([kf_v.get_cart(2)], dims)).^2, 2)./var_vy';


%% Figure
figure(1)
plot(1:N,y_hat(1,:,1), 'r')
hold off

%% The error
e1 = (y_hat-y).^2;
e1m = mean(e1,3);
e2 = zeros(1,N,M);
for m = 1:M
    for n = 1:N
        e2(1,n,m) = norm(kf_xp(:,n,m)-u);
    end    
end
e2m = mean(e2,3);

figure(2)
plot(1:N,10*log10(e1m),'b')
title('MSE: Kalman Filter.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-40 0])
grid on

figure(3)
plot(1:N,20*log10(e2m),'r')
title('MSD: Kalman Filter.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-40 0])
grid on

% %% Print Trajectories (Scatter)

% figure(2)
% % rx
% subplot(2,2,1)
% hold on;
% plot(10*log10(obs_mse_rx), '-k', 'LineWidth', 1.2);
% plot(10*log10(kf_mse_rx), '--g', 'LineWidth', 1.2);
% hold off;
% grid on;
% title('MSE r_x [dB]')
% legend('obs', 'KF');
% set(gca, 'YLim', [-20, 5], 'XLim', [0 200]);
% % vx
% subplot(2,2,2)
% hold on;
% plot(10*log10(obs_mse_vx), '-k', 'LineWidth', 1.2);
% plot(10*log10(kf_mse_vx), '--g', 'LineWidth', 1.2);
% hold off;
% grid on;
% title('MSE v_x [dB]')
% legend('obs', 'KF');
% set(gca, 'YLim', [-60, 5]);
% % ry
% subplot(2,2,3)
% hold on;
% plot(10*log10(obs_mse_ry), '-k', 'LineWidth', 1.2);
% plot(10*log10(kf_mse_ry), '--g', 'LineWidth', 1.2);
% hold off;
% grid on;
% title('MSE r_y [dB]')
% legend('obs', 'KF');
% set(gca, 'YLim', [-20, 5], 'XLim', [0 200]);
% % vx
% subplot(2,2,4)
% hold on;
% plot(10*log10(obs_mse_vy), '-k', 'LineWidth', 1.2);
% plot(10*log10(kf_mse_vy), '--g', 'LineWidth', 1.2);
% hold off;
% grid on;
% title('MSE v_y [dB]')
% legend('obs', 'KF');
% set(gca, 'YLim', [-60, 5]);


% figure(1)
% subplot(2,1,1)
% plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')
% hold on
% scatter([ship_measure(:,m).get_cart(1)], [ship_measure(:,m).get_cart(2)], 'black', '*')
% plot([kf_r(:,m).get_cart(1)], [kf_r(:,m).get_cart(2)], '-^g', 'LineWidth', 1.5, 'MarkerSize', 1.5)
% hold off
% grid on
% subplot(2,1,2)
% plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')
% hold on
% scatter([ship_measure(:,m).get_cart(1)], [ship_measure(:,m).get_cart(2)], 'black', '*')
% plot([kf_r(:,m).get_cart(1)], [kf_r(:,m).get_cart(2)], '-^g', 'LineWidth', 1.5, 'MarkerSize', 1.5)
% hold off
% set(gca,'YLim', [0 8e3])
% grid on

% figure(4)
% subplot(2,1,1)
% plot(timestamp, [kf_v(:,m).get_cart(1)], '--g', 'LineWidth', 1.5)
% set(gca, 'YLim', [-10 10])
% hold off
% grid on
% subplot(2,1,2)
% plot(timestamp, [kf_v(:,m).get_cart(2)], '--g', 'LineWidth', 1.5)
% set(gca, 'YLim', [-10 10])
% hold off
% grid on