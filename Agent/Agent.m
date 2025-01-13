classdef Agent < handle
    properties
        y_hat               % Variável de observação que se estima a partir de H
        H_hat               % Variável de pesos que estima
        x_dim               % Dimensão do vetor estado
        y_dim               % Dimensão do vetor observação
        n_window            % Tamanho da janela temporal do agente
        agent_technique     % Técnica de update adaptativo
        % neighbors           % Matriz de peso de Agentes vizinhos
        obs_buffer          % Janela temporal de observação do agente
        state_buffer        % Janela tempora de estados do agente
    end

    methods
        function obj = Agent(varargin)
            p = inputParser;
            
            default_x_dim = 3;
            check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_y_dim = 1;
            check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            default_n_win = 10;
            check_n_win = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1) == 0);

            % default_x_ini = zeros(default_x_dim, 1);
            % check_x_ini = @(x) isnumeric(x);

            % default_agent_tech = Wiener('x_dim', default_x_dim, 'y_dim', default_y_dim, 'n_win', default_n_win);
            default_agent_tech = Wiener();
            check_agent_tech = @(x) isa(x, "Agent_technique");

            addOptional(p, 'x_dim', default_x_dim, check_x_dim);
            addOptional(p, 'y_dim', default_y_dim, check_y_dim);
            addOptional(p, 'n_win', default_n_win, check_n_win);
            addOptional(p, 'agent_tech', default_agent_tech, check_agent_tech);
            % addOptional(p, 'x_ini', default_x_ini, check_x_ini);

            try
                parse(p, varargin{:});
                obj.x_dim = p.Results.x_dim;
                obj.y_dim = p.Results.y_dim;
                obj.n_window = p.Results.n_win;

                % if (size(p.Results.x_ini, 1) ~= obj.x_dim) && (size(p.Results.x_ini, 2) > 1)
                %     error('The size of "x_ini" must be equal to the "x_dim".');
                % else
                %     obj.y_hat = p.Results.x_ini;
                % end

                % if p.Results.agent_tech.get_win_length() ~= obj.n_window
                %     error('The length of "agent_technique" window must be equal to "n_win".');
                % else
                %     obj.agent_technique = p.Results.agent_tech;
                % end
                obj.agent_technique = p.Results.agent_tech;     % Faz com o número de amostras que enviar para função apply

                % Caso o usuario tenha colocado 'x_dim' e 'agent_tech', verifica as dimensoes
                if ~any(strcmp('x_dim', p.UsingDefaults)) && ~any(strcmp('agent_tech', p.UsingDefaults)) && (obj.x_dim ~= obj.agent_technique.x_dim)
                    error('The "x_dim" of "agent_tech" must be equal to the "x_dim" declared.');

                % Caso o usuario tenha colocado 'y_dim' e 'agent_tech', verifica as dimensoes
                elseif ~any(strcmp('y_dim', p.UsingDefaults)) && ~any(strcmp('agent_tech', p.UsingDefaults)) && (obj.y_dim ~= obj.agent_technique.y_dim)
                    error('The "y_dim" of "agent_tech" must be equal to the "y_dim" declared.');

                % Caso o usuario tenha colocado 'n_win' e 'agent_tech', verifica as dimensoes
                elseif ~any(strcmp('n_win', p.UsingDefaults)) && ~any(strcmp('agent_tech', p.UsingDefaults)) && (obj.y_dim ~= obj.agent_technique.n_win)
                    error('The "n_win" of "agent_tech" must be equal to the "n_win" declared.');

                elseif any(strcmp('agent_tech', p.UsingDefaults)) && (~any(strcmp('x_dim', p.UsingDefaults)) || ~any(strcmp('y_dim', p.UsingDefaults))  || ~any(strcmp('n_win', p.UsingDefaults)))
                    obj.agent_technique = Wiener('x_dim', obj.x_dim, 'y_dim', obj.y_dim, 'n_win', obj.n_window);
                end
                
                obj.H_hat = obj.agent_technique.get_H();
                obj.y_hat = zeros(obj.y_dim, 1);
                obj.obs_buffer = zeros(obj.y_dim, obj.n_window);
                obj.state_buffer = zeros(obj.x_dim, obj.n_window);
                
            catch exception
                error('An error occurred %s\n', exception.message);
            end
        end

        function obj = self_learning_step(obj, st, obs) % x, d from Diniz, Adaptive Filtering
            obj.obs_buffer = obj.update_buffer(obj.obs_buffer, obs);
            obj.state_buffer = obj.update_buffer(obj.state_buffer, st);
            obj.y_hat = obj.agent_technique.apply(obj.obs_buffer, obj.state_buffer);
            obj.H_hat = obj.agent_technique.get_H();
        end
        
        function obj = social_learning_step(obj, new_H_hat)
            obj.H_hat = new_H_hat;  % Agente sem vontade própria
            obj.agent_technique.update_H(obj.H_hat);
        end
        
        function y_hat = get_y_hat(obj)
            obj.update_y_hat();
            y_hat = obj.y_hat;
        end

        function obj = update_y_hat(obj)
            obj.y_hat = obj.agent_technique.get_y_hat(obj.state_buffer(:,1));
        end

        function H_hat = get_H_hat(obj)
            H_hat = obj.H_hat;
        end

        function obj = reset(obj)
            obj.H_hat = zeros(size(obj.H_hat));
            obj.obs_buffer = zeros(size(obj.obs_buffer));
            obj.state_buffer = zeros(size(obj.state_buffer));
            obj.y_hat = zeros(size(obj.y_hat));
            obj.agent_technique.reset();
        end
        
    end
    methods (Static)
        function buf = update_buffer(buffer, new)   % Primitives
            buf = buffer;
            buf(:,2:end) = buf(:,1:end-1);
            buf(:,1) = new;
        end
    end
end