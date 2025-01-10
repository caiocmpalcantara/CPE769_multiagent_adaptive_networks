% TEST THE BASIC STRUCTURE OF KALMAN CLASS SIMULATION
%   > Using Utils: GeoPoint, Displacement, Speed, Noise Model
%   > Using Technique: Kalman, KF
%   > Using Technique/Kalman_inc: State Model
addpath('./Utils/', ...
        './Technique/', ...
        './Technique/Kalman_inc')

%% Simulate a Trajectory
% Position
ownShipPoint = GeoPoint(-24,-41);
% Target
r0 = 5e3*sqrt(2);
phi0 = pi-pi/4;
targetSub = GeoPoint(r0, phi0, 'Polar', ownShipPoint);
v0 = Speed(5, 0, 'Polar');
% Vectors
target_history(1) = targetSub;
d(1) = target_history(1) - ownShipPoint;
ship_measure(1) = d(1);
timestamp(1) = datetime('now');
% Simulation
T = 10;     %[s]
N = 200;    %2000 segundos no total
M = 100;    %Monte-Carlo
f_medium = 15e3;
f_signal = 16e3;
range_sd = 15;  %[m]
bearing_sd = 2; %[deg]
a_fc_R = 1/2*1e-3;
a_fc_phi = 1/4*1e-3;
nu_R = Noise_Simple(f_medium, a_fc_R, range_sd);
nu_phi = Noise_Simple(f_medium, a_fc_phi, bearing_sd);
% nu_R = Noise_Simple(f_medium);
% nu_phi = Noise_Simple(f_medium, 1e-6*deg2rad(1), deg2rad(2));
for n = 2:N
    target_history(n) = target_history(n-1) + v0 .* T;
    d(n) = target_history(n) - ownShipPoint;
    timestamp(n) = timestamp(n-1) + seconds(T);
end
figure(1)
plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')

%% Kalman Filtering
% Kalman Init
range_sd = 15;  %[m]
bearing_sd = 2; %[deg]

dims = [N M];
ship_measure = Displacement.array(dims);
%
kf_r = Displacement.array(dims);
kf_v = Speed.array(dims);

% Monte-Carlo: Kalman Processing (Update new measurement)
rng(8988466);
for m = 1:M
    ship_measure(1,m) = Displacement(d(1).get_cart(1)+nu_R.rand(f_signal), d(1).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
    ship_measure(2,m) = Displacement(d(2).get_cart(1)+nu_R.rand(f_signal), d(2).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
    kf = KF(ship_measure(1,m), ship_measure(2,m), timestamp(1), timestamp(2), range_sd, deg2rad(bearing_sd));
   
    kf_r(1,m) = ship_measure(1,m);
    kf_r(2,m) = ship_measure(2,m);
    kf_v(1,m) = Speed(0,0,'Cartesian');
    kf_v(2,m) = kf.posterior_estimate_speed();
   
    for n = 3:N
        ship_measure(n,m) = Displacement(d(n).get_cart(1)+nu_R.rand(f_signal), d(n).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
        %
        kf.update(ship_measure(n,m), timestamp(n));
        kf_r(n,m) = kf.posterior_estimate_position();
        kf_v(n,m) = kf.posterior_estimate_speed();
    end
end

%% Monte-Carlo sums
% Position-Dependent variance
var_rx = range_sd^2*ones(1,N);
var_ry = range_sd^2*ones(1,N);
var_vx = (2*var_rx)./(seconds(timestamp(2)-timestamp(1)))^2;
var_vy = (2*var_ry)./(seconds(timestamp(2)-timestamp(1)))^2;

% MSE Noisy obs 
% reshape([pdnkf_v.get_cart(2)], dims)
obs_mse_rx = mean((d.get_cart(1)' * ones(1,M) - reshape([ship_measure.get_cart(1)], dims)).^2, 2)./var_rx';
obs_mse_ry = mean((d.get_cart(2)' * ones(1,M) - reshape([ship_measure.get_cart(2)], dims)).^2, 2)./var_ry';
vx = [ones(1,M); reshape([ship_measure(2:end,:).get_cart(1) - ship_measure(1:end-1,:).get_cart(1)], [N-1,M])/T];
obs_mse_vx = mean((ones(N,M) * v0.get_cart(1)- vx).^2, 2)./var_vx';
vy = [ones(1,M); reshape([ship_measure(2:end,:).get_cart(2) - ship_measure(1:end-1,:).get_cart(2)], [N-1,M])/T];
obs_mse_vy = mean((ones(N,M) * v0.get_cart(2)- vy).^2, 2)./var_vy';

% MSE KF
kf_mse_rx = mean((d.get_cart(1)' * ones(1,M) - reshape([kf_r.get_cart(1)], dims)).^2, 2)./var_rx';
kf_mse_ry = mean((d.get_cart(2)' * ones(1,M) - reshape([kf_r.get_cart(2)], dims)).^2, 2)./var_ry';
kf_mse_vx = mean((ones(N,M) * v0.get_cart(1) - reshape([kf_v.get_cart(1)], dims)).^2, 2)./var_vx';
kf_mse_vy = mean((ones(N,M) * v0.get_cart(2) - reshape([kf_v.get_cart(2)], dims)).^2, 2)./var_vy';


%% Print Trajectories (Scatter)
figure(2)
% rx
subplot(2,2,1)
hold on;
plot(10*log10(obs_mse_rx), '-k', 'LineWidth', 1.2);
plot(10*log10(kf_mse_rx), '--g', 'LineWidth', 1.2);
hold off;
grid on;
title('MSE r_x [dB]')
legend('obs', 'KF');
set(gca, 'YLim', [-20, 5], 'XLim', [0 200]);
% vx
subplot(2,2,2)
hold on;
plot(10*log10(obs_mse_vx), '-k', 'LineWidth', 1.2);
plot(10*log10(kf_mse_vx), '--g', 'LineWidth', 1.2);
hold off;
grid on;
title('MSE v_x [dB]')
legend('obs', 'KF');
set(gca, 'YLim', [-60, 5]);
% ry
subplot(2,2,3)
hold on;
plot(10*log10(obs_mse_ry), '-k', 'LineWidth', 1.2);
plot(10*log10(kf_mse_ry), '--g', 'LineWidth', 1.2);
hold off;
grid on;
title('MSE r_y [dB]')
legend('obs', 'KF');
set(gca, 'YLim', [-20, 5], 'XLim', [0 200]);
% vx
subplot(2,2,4)
hold on;
plot(10*log10(obs_mse_vy), '-k', 'LineWidth', 1.2);
plot(10*log10(kf_mse_vy), '--g', 'LineWidth', 1.2);
hold off;
grid on;
title('MSE v_y [dB]')
legend('obs', 'KF');
set(gca, 'YLim', [-60, 5]);


figure(1)
subplot(2,1,1)
plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')
hold on
scatter([ship_measure(:,m).get_cart(1)], [ship_measure(:,m).get_cart(2)], 'black', '*')
plot([kf_r(:,m).get_cart(1)], [kf_r(:,m).get_cart(2)], '-^g', 'LineWidth', 1.5, 'MarkerSize', 1.5)
hold off
grid on
subplot(2,1,2)
plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')
hold on
scatter([ship_measure(:,m).get_cart(1)], [ship_measure(:,m).get_cart(2)], 'black', '*')
plot([kf_r(:,m).get_cart(1)], [kf_r(:,m).get_cart(2)], '-^g', 'LineWidth', 1.5, 'MarkerSize', 1.5)
hold off
set(gca,'YLim', [0 8e3])
grid on

figure(4)
subplot(2,1,1)
plot(timestamp, [kf_v(:,m).get_cart(1)], '--g', 'LineWidth', 1.5)
set(gca, 'YLim', [-10 10])
hold off
grid on
subplot(2,1,2)
plot(timestamp, [kf_v(:,m).get_cart(2)], '--g', 'LineWidth', 1.5)
set(gca, 'YLim', [-10 10])
hold off
grid on