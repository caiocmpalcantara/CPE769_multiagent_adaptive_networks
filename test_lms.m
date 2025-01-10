addpath("./Technique/")
rng(8988466)
x = [1 1 1]';
u = [1 1 1];
N = 200;
n = 1:N;
noise = randn(1,N);

d = u*x + noise;
%% Figures
figure(1)
plot(n,d,'b')
xlabel('Time')
ylabel('Observation')
title('Testing LMS Class')
hold on
grid on

%% Criando a técnica
tec1 = [Lms(); ...
        Lms('H_ini', [0 0 0]); ...
        Lms('n_win', 50); ...
        Lms('mu', 0.5); ...
        Lms('epsilon', 0.1)];
buf_obs = zeros(tec1(2).y_dim, tec1(2).n_win);
buf_st = zeros(tec1(2).x_dim, tec1(2).n_win);
y_hat = zeros(tec1(2).y_dim, N);
u_hat = zeros(tec1(2).y_dim, tec1(2).x_dim, N);
for i = 1:N
    buf_obs = update_buffer(buf_obs, d(i));
    buf_st = update_buffer(buf_st, x);
    y_hat(i) = tec1(2).apply(buf_obs, buf_st);
    u_hat(:,:,i) = tec1(2).get_H();
end

plot(n, y_hat, 'r')
hold off

%% The error
e1 = (y_hat-u*x).^2;
e2 = zeros(1,N);
for i = 1:N
    e2(i) = norm(u_hat(:,:,i)-u);
end
figure(2)
plot(n,10*log10(e1),'b')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on

figure(3)
plot(n,e2,'r')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on

%% functions
function buff = update_buffer(buffer, new_sample)
    buff = buffer;
    buff(:, 2:end) = buff(:, 1:end-1);
    buff(:, 1) = new_sample;
end