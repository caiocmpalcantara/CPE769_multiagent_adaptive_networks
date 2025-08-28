classdef Agent_vector < handle
    properties
        coop_type           % Cooperation type
        n_agents            % Number of agents in the vector
        agents_vec          % Agents vector
        B                   % Matrix topology (para cada agente, uma coluna)
        x_dim               % Dimension of state vector x
        y_dim               % Dimension of observation vector y
        fusion_technique    % Técnica da fusão entre vizinhos
    end

    methods
        function obj = Agent_vector(varargin)
            p = inputParser;

            default_coop_type = Cooperation_type.non_cooperative;
            check_coop_type = @(x) isa(x,"Cooperation_type");

            default_n_agents = 3;
            check_n_agents = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_agents_vec = repmat(Agent(), [1, default_n_agents]);
            for a = 1:default_n_agents
                default_agents_vec(a) = Agent();
            end
            check_agents_vec = @(x) isa(x, "Agent");

            default_B = eye(default_n_agents);
            check_B = @(x) isnumeric(x);

            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_y_dim = 1;
            check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            addOptional(p, 'coop_type', default_coop_type, check_coop_type);
            addOptional(p, 'n_agents', default_n_agents, check_n_agents);
            addOptional(p, 'agents_vec', default_agents_vec, check_agents_vec);
            addOptional(p, 'B_matrix', default_B, check_B);
            addOptional(p, 'x_dim', default_x_dim, check_x_dim);
            addOptional(p, 'y_dim', default_y_dim, check_y_dim);

            try
                parse(p, varargin{:});
                obj.coop_type = p.Results.coop_type;
                if any(strcmp('B_matrix', p.UsingDefaults)) && ~any(strcmp('n_agents', p.UsingDefaults))
                    obj.B = eye(p.Results.n_agents);
                    obj.n_agents = p.Results.n_agents;
                elseif ~any(strcmp('B_matrix', p.UsingDefaults)) && any(strcmp('n_agents', p.UsingDefaults))
                    obj.B = p.Results.B_matrix;
                    obj.n_agents = size(p.Results.B_matrix, 1);
                else
                    obj.B = p.Results.B_matrix;
                    obj.n_agents = p.Results.n_agents;
                end
                if (size(obj.B,1) ~= obj.n_agents) && (size(obj.B,2) ~= obj.n_agents)
                    error('The size of "B" must be compatible with the "n_agents".')
                end
                if any(strcmp('agents_vec', p.UsingDefaults)) && (~any(strcmp('n_agents', p.UsingDefaults)) || ~any(strcmp('B_matrix', p.UsingDefaults)))
                    obj.agents_vec = repmat(Agent(), [1, obj.n_agents]);
                    for a = 1:obj.n_agents
                        obj.agents_vec(a) = Agent();
                    end
                else
                    obj.agents_vec = p.Results.agents_vec;
                end
                if (length(obj.agents_vec) ~= obj.n_agents)
                    error('The size of "agents_vec" must be compatible with the "n_agents".')
                end
                obj.x_dim = p.Results.x_dim;
                obj.y_dim = p.Results.y_dim;

                switch obj.coop_type
                    case Cooperation_type.non_cooperative
                        obj.fusion_technique = Non_cooperative(obj);

                    case Cooperation_type.consensus_constrain
                        obj.fusion_technique = Consensus_constrain(obj);

                    case Cooperation_type.single_task
                        obj.fusion_technique = Single_task(obj);

                    case Cooperation_type.multi_task
                        obj.fusion_technique = Multi_task(obj);
                        
                    otherwise
                        error('The "coop_type" must be "Cooperation_type".');
                end

            catch exception
                error('An error occurred: %s\n', exception.message);
            end
        end
        
        % function [obj, H_hat_history] = simulate(obj, state_dynamics, observations, n_samples)
        %     H_hat_history = zeros(obj.y_dim, obj.x_dim, obj.n_agents, n_samples);
        %     for n = 1:n_samples
        %         obj.self_learning_step(state_dynamics(:,a,n), observations(:,a,n));
        %         obj.social_learning_step();
        %         H_hat_history(:,:,:,n) = obj.collec_H_hat();
        %     end
        % end
        % size(obs_per_agent) = (dim_y, n_agents)
        function obj = update(obj, obs_per_agent, state)
            obj.self_learning_step(state, obs_per_agent);
            obj.social_learning_step();
        end

        function obj = self_learning_step(obj, st, obs_per_agent)
            for a = 1:obj.n_agents
                obj.agents_vec(a) = obj.agents_vec(a).self_learning_step(st, obs_per_agent(:,a));
            end
        end
        
        function obj = social_learning_step(obj)
            collec_H_hat = obj.collec_H_hat();
            obj.fusion_technique.apply(collec_H_hat, obj.B);
        end
        
        function collec_H_hat = collec_H_hat(obj)   %FIXME: O agent vec acaba funcionando como uma central (na forma como foi programado)
            collec_H_hat = zeros(obj.y_dim, obj.x_dim, obj.n_agents);
            for a = 1:obj.n_agents
                collec_H_hat(:,:,a) = obj.agents_vec(a).get_H_hat();
            end
        end

        function n_agents = get_n_agents(obj)
            n_agents = obj.n_agents;
        end

        function obj = reset(obj)
            for a = 1:obj.n_agents
                obj.agents_vec(a).reset();
            end
        end
    end
end