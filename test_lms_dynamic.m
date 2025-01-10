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

x = [[d.get_cart(1)]; [d.get_cart(2)]; v0.get_cart(1)*ones(1,N); v0.get_cart(2)*ones(1,N)];

%% Figures
figure(1)
plot(x(1,:), x(2,:), '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')
xlabel('r_x')
ylabel('r_y')
title('Dynamics Scatter')
hold on
grid on

%% Criando a técnica + Simulation
tec1 = Lms('x_dim', 4, 'y_dim', 4, 'H_ini',[1 0 10 0; 0 1 0 10; 0 0 1 0; 0 0 0 1],'epsilon', 1e3, 'mu', 5.5e-9, 'n_win', 10);

buf_obs = zeros(tec1.y_dim, tec1.n_win);
buf_st = zeros(tec1.x_dim, tec1.n_win);
y = zeros(tec1.y_dim, N, M);
u_hat = zeros(tec1.y_dim, tec1.x_dim, N, M);
x_obs = zeros(tec1.y_dim, N, M);

% Init
range_sd = 15;  %[m]
bearing_sd = 2; %[deg]
speed_sd = 0.2; %[m/s]

dims = [N M];

A = [1 0 T 0; 0 1 0 T; 0 0 1 0; 0 0 0 1];
nu_x = [range_sd; range_sd; speed_sd; speed_sd];
rng(8988466);

% x_{n} = A*x_{n-1} + nu_x

for m = 1:M
    for i = 1:N
        % if i == 1
        %     x_obs(:,i,m) = x(:,i) + nu_x .* randn(tec1.x_dim, 1);
        % else
        %     x_obs(:,i,m) = A*x(:,i-1) + nu_x .* randn(tec1.x_dim, 1);
        % end
        x_obs(:,i,m) = x(:,i) + nu_x .* randn(tec1.x_dim, 1);
        
        if i == 1
            buf_obs = update_buffer(buf_obs, x_obs(:,i,m));
            buf_st = update_buffer(buf_st, x_obs(:,i,m));
            y(:,i,m) = tec1.apply(buf_obs, buf_st);
            u_hat(:,:,i,m) = tec1.get_H();
        else
            buf_obs = update_buffer(buf_obs, x_obs(:,i,m));
            buf_st = update_buffer(buf_st, u_hat(:,:,i-1,m)*y(:,i-1,m));
            y(:,i,m) = tec1.apply(buf_obs, buf_st);
            u_hat(:,:,i,m) = tec1.get_H();
        end
    end    
end

plot(x_obs(1,:,end), x_obs(2,:,end), '-xk', 'LineWidth', 1.5, 'MarkerFaceColor', 'black')
plot(y(1,:,end), y(2,:,end), '--b', 'LineWidth', 1.5)
hold off

%% The error
e1 = (y-x).^2;
e1m = mean(e1,3);
figure(2)
plot(n, 10*log10(norm(e1m(1,:),e1m(2,:),2)), 'b')
title('MSE')
ylabel('e[n]')
xlabel('n')
grid on
hold off

%% functions
function buff = update_buffer(buffer, new_sample)
    buff = buffer;
    buff(:, 2:end) = buff(:, 1:end-1);
    buff(:, 1) = new_sample;
end