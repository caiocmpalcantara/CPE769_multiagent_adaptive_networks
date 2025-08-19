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
title('Testing RLS Class')
hold on
grid on

%% Criando a técnica
tec1 = [Rls() Rls('H_ini', [0 0 0]) Rls('lambda', .85) Rls('delta', 0.8)];
A = length(tec1);
a=3;
buf_obs = zeros(tec1(a).y_dim, 5);
buf_st = zeros(tec1(a).x_dim, 5);
y = zeros(tec1(a).y_dim, N);
u_hat = zeros(tec1(a).y_dim, tec1(a).x_dim, N);
for i = 1:N
    buf_obs = update_buffer(buf_obs, d(i));
    buf_st = update_buffer(buf_st, x);
    y(i) = tec1(a).apply(buf_obs, buf_st);
    u_hat(:,:,i) = tec1(a).get_H();
end

plot(n, y, 'r')
hold off

%% The error: absolut
e1 = abs(y-u*x);
for i = 1:N
    e2(i) = norm(u_hat(:,:,i)-u);
end
figure(2)
plot(n,10*log10(e1),'b')
hold on
plot(n,20*log10(e2),'r')
hold off
grid on
title('Test RLS: Absolut and Norm Error.')
ylabel('e[n]')
xlabel('n')
legend('abs', 'nor')
%% functions
function buff = update_buffer(buffer, new_sample)
    buff = buffer;
    buff(:, 2:end) = buff(:, 1:end-1);
    buff(:, 1) = new_sample;
end