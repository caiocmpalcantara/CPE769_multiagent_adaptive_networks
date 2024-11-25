classdef Agent_vector
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

            default_coop_type = Cooperation_type.Non_cooperative;
            check_coop_type = @(x) isa(x,"Cooperation_type");

            default_n_agents = 3;
            check_n_agents = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_agents_vec = repmat(Agent(), [default_n_agents, 1]);
            check_agents_vec = @(x) isa(x, "Agent");

            default_B = eye(default_n_agents);
            check_B = @(x) isnumeric(x);

            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);



            addOptional(p, 'coop_type', default_coop_type, check_coop_type);
            addOptional(p, 'n_agents', default_n_agents, check_n_agents);
            addOptional(p, 'agent_vec', default_agents_vec, check_agents_vec);
            addOptional(p, 'B_matrix', default_B, check_B);
            addOptional(p, 'x_dim', default_x_dim, check_x_dim);

            try
                parse(p, varargin{:});
                obj.coop_type = p.Results.coop_type;
                obj.n_agents = p.Results.n_agents;
                if (size(p.Results.B_matrix,1) ~= obj.n_agents) && (size(p.Results.B_matrix,2) ~= obj.n_agents)
                    error('The size of "B" must be compatible with the "n_agents".')
                end
                obj.B = p.Results.B_matrix;
                obj.agents_vec = p.Results.agent_vec;
                obj.x_dim = p.Results.x_dim;

                switch obj.coop_type
                    case Cooperation_type.non_cooperative
                        obj.fusion_technique = Non_cooperative(obj.B);

                    case Cooperation_type.consensus_constrain
                        obj.fusion_technique = Consensus_constrain(obj.B);

                    case Cooperation_type.single_task
                        obj.fusion_technique = Single_task(obj.B);

                    case Cooperation_type.multi_task
                        obj.fusion_technique = Multi_task(obj.B);
                        
                    otherwise
                        error('The "coop_type" must be "Cooperation_type".');
                end

            catch exception
                error('An error occurred: %s\n', exception.message);
            end
        end
        
        function H_hat_history = simulate(obj, state_dynamics, observations, n_samples)
            H_hat_history = zeros(obj.y_dim, obj.x_dim, obj.n_agents, n_samples);
            for n = 1:n_samples
                obj.self_learning_step(state_dynamics(:,a,n), observations(:,a,n));
                obj.social_learning_step();
                H_hat_history(:,:,:,n) = obj.collec_H_hat();
            end
        end

        function self_learning_step(obj, obs)
            for a = 1:obj.n_agents
                obj.agents_vec(a).self_learning_step(obs);
            end
        end
        
        function social_learning_step(obj)
            collec_H_hat = obj.collec_H_hat();
            obj.fusion_technique.apply(obj.agents_vec, collec_H_hat, obj.B);
        end
        
        function collec_H_hat = collec_H_hat(obj)
            collec_H_hat = zeros(obj.y_dim, obj.x_dim, obj.n_agents);
            for a = 1:obj.n_agents
                collec_H_hat(:,:,a) = obj.agents_vec(a).get_H_hat();
            end
        end

        function n_agents = get_n_agents(obj)
            n_agents = obj.n_agents;
        end
    end
end