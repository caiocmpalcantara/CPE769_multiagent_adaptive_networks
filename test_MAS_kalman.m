% Initial test Multi-Agent System
addpath("./Technique/")
addpath("./Technique/Kalman_inc/")
addpath("./Agent/")
addpath("./Utils/")

%% Simulation: Trajectory
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
% nu_phi = Noise_Simple(f_medium, a_fc_phi, bearing_sd);
% nu_R = Noise_Simple(f_medium);
% nu_phi = Noise_Simple(f_medium, 1e-6*deg2rad(1), deg2rad(2));
for n = 2:N
    target_history(n) = target_history(n-1) + v0 .* T;
    d(n) = target_history(n) - ownShipPoint;
    timestamp(n) = timestamp(n-1) + seconds(T);
end
figure(1)
plot([d.get_cart(1)], [d.get_cart(2)], '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'red')

%% Simulation: Monte-Carlo
rng(8988466);

% Init
range_sd = 15;  %[m]
bearing_sd = 2; %[deg]

Na = 6;
dims = [N M Na];
ship_measure = Displacement.array(dims);
%
kf_r = Displacement.array(dims);
kf_v = Speed.array(dims);

a_vec = repmat(Agent(), [6, 1]);

for m = 1:M
    for a = 1:Na
        ship_measure(1,m,a) = Displacement(d(1).get_cart(1)+nu_R.rand(f_signal), d(1).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
        ship_measure(2,m,a) = Displacement(d(2).get_cart(1)+nu_R.rand(f_signal), d(2).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
        kf = KF(ship_measure(1,m,a), ship_measure(2,m,a), timestamp(1), timestamp(2), range_sd, deg2rad(bearing_sd));
    
        kf_r(1,m,a) = ship_measure(1,m,a);
        kf_r(2,m,a) = ship_measure(2,m,a);
        kf_v(1,m,a) = Speed(0,0,'Cartesian');
        kf_v(2,m,a) = kf.posterior_estimate_speed();

        a_vec(a) = Agent('agent_tech', KF(ship_measure(1,m,a), ship_measure(2,m,a), timestamp(1), timestamp(2), range_sd, deg2rad(bearing_sd)));
    end
    % Create Kalman agents
    % Choose an Agent vector and Run sim
    agent_vec = Agent_vector('n_agents', Na, 'agent_vec', a_vec);
    for n = 3:N
        ship_measure(n,m) = Displacement(d(n).get_cart(1)+nu_R.rand(f_signal), d(n).get_cart(2)+nu_R.rand(f_signal), 'Cartesian');
        %
        kf.update(ship_measure(n,m), timestamp(n));
        kf_r(n,m) = kf.posterior_estimate_position();
        kf_v(n,m) = kf.posterior_estimate_speed();
    end

end

%% Prints

