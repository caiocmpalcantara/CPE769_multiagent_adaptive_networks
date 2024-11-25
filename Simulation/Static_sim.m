classdef Static_sim < sim
    properties
        noise       % Noise type
        x           % State
        H           % Observation matrix => u
        A           % State transition matrix
        x_dim       % Dimension of state vector
        y_dim       % Dimension of observation vector
        n_agents    % Number of agents
        n_samples   % Number of samples
        agent_vec   % Agent vector
    end

    methods
        function obj = sim(varargin)
            p = inputParser;

            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_y_dim = 1;
            check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_x = [1 1 1]';
            check_x = @(x) isnumeric(x);

            default_H = [1 1 1];
            check_H = @(x) isnumeric(x);

            default_n_agents = 3;
            check_n_agents = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_n_samples = 50;
            check_n_samples = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_noise = N_Gaussian();
            check_noise = @(x) isa(x, "Noise");

            default_agent_vec = Agent_vector('n_agents', default_n_agents);
            check_agent_vec = @(x) isa(x, "Agent_vector");

            addOptional(p, 'x_dim', default_x_dim, check_x_dim);
            addOptional(p, 'y_dim', default_y_dim, check_y_dim);
            addOptional(p, 'state', default_x, check_x);
            addOptional(p, 'obs_matrix', default_H, check_H);
            addOptional(p, 'n_agents', default_n_agents, check_n_agents);
            addOptional(p, 'n_samples', default_n_samples, check_n_samples);
            addOptional(p, 'noise', default_noise, check_noise);
            addOptional(p, 'agent_vec', default_agent_vec, check_agent_vec);

            try
                parse(p, varargin{:});
                % Check compatibility between 'x' and 'x_dim'
                if ~isempty(p.Results.x_dim) && ~isempty(p.Results.x) && (length(p.Results.x) ~= p.Results.x_dim)
                    error('The length of "x" must match the value of "x_dim".');
                elseif size(p.Results.obs_matrix, 2) ~= p.Results.x_dim
                    error('The number of columns of "H" must match to "x_dim".');
                elseif size(p.Results.obs_matrix, 1) ~= p.Results.y_dim
                    error('The number of rows of "H" must match to "y_dim".');
                end
                % Check compatibility between 'n_agents' and 'agent_vec' length
                if ~isempty(p.Results.n_agents) && ~isempty(p.Results.agent_vec) && (p.Results.agent_vec.get_n_agents() ~= p.Results.n_agents)
                    error('The number of agents in "agent_vec" must match to "n_agents".')
                end

                obj.x_dim = p.Results.x_dim;
                obj.x = p.Results.state;
                obj.H = p.Results.obs_matrix;
                obj.n_agents= p.Results.n_agents;
                obj.n_samples = p.Results.n_samples;
                obj.noise = p.Results.noise;
                obj.A = eye(obj.x_dim);
                obj.agent_vec = p.Results.agent_vec;
                
            catch exception
                error('An error occurred: %s\n', exception.message);
            end
        end
        function [state_dynamics, observations] = simulate_obs(obj) % Replicar equação de target
            noise = obj.noise.realize([obj.y_dim, obj.n_agents, obj.n_samples]);

            observations = zeros(obj.y_dim, obj.n_agents, obj.n_samples);
            state_dynamics = zeros(obj.x_dim, obj.n_agents, obj.n_samples);
            for n = 1:obj.n_samples
                for k = 1:obj.n_agents
                    observations(:,k,n) = obj.H * obj.x;
                    state_dynamics(:,k,n) = obj.A * obj.x;
                end
                %no state transition
                obj.x = obj.A*obj.x;
            end

            observations = observations + noise;
            % Equação Target Global (todos os instantes de tempo e todos os targets)
            % Yn = Hn X + Vnc
        end
        function H_hat_history = simulate_agents(obj, state_dynamics, observations)
            
            % Then simulate they for each observation
            H_hat_history = obj.agent_vec.simulate(state_dynamics, observations, obj.n_samples);
        end
    end
end