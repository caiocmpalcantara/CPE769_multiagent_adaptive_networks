classdef Lms < Agent_technique
    properties
        H                   % matrix de pesos atual
        % n_win               % tamanho da janela de observação
        mu                  % passo adaptativo
        iteracts            % número de iterações
        % R_hat               % Para estimar mu otimo variável 
        % state_buffer        % buffer de estado
        % n_state_buf_len     % tamanho do buffer de estados
        epsilon             % termo de regularização
    end

    methods
        function obj = Lms(varargin)
            % disp(varargin)
            % Parametros da Classe derivada
            % derivedParams = {'order', 'mu', 'H_ini', 'H_rnd_ini', 'epsilon'};
            derivedParams = {'mu', 'H_ini', 'H_rnd_ini', 'epsilon'};

            % Separação entre parâmetros da classe derivada e base
            isderivedParam = ismember(varargin(1:2:end), derivedParams);
            % para capturar tanto o arg_label, quanto o arg_value
            baseArgs = varargin(~reshape([isderivedParam; isderivedParam], 1, []));
            derivedArgs = varargin(reshape([isderivedParam; isderivedParam], 1, []));
            % disp(baseArgs)

            obj@Agent_technique(baseArgs{:});
           
            p = inputParser;

            % default_n_win = 5;
            % check_n_win = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);

            default_mu = 0.1;
            check_mu = @(x) isnumeric(x) && isscalar(x) && (x > 0);

            default_H_ini = ones(obj.y_dim, obj.x_dim);
            check_H_ini = @(x) isnumeric(x);

            default_H_rnd_ini = false;
            check_H_rnd_ini = @(x) islogical(x);

            % default_n_state_buf_len = 1;
            % check_n_state_buf_len = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);

            default_epsilon = 0;
            check_epsilon = @(x) isnumeric(x) && isscalar(x) && (x > 0);

            % addOptional(p, 'order', default_n_win, check_n_win);
            addOptional(p, 'mu', default_mu, check_mu);
            addOptional(p, 'H_ini', default_H_ini, check_H_ini);
            addOptional(p, 'H_rnd_ini', default_H_rnd_ini, check_H_rnd_ini);
            % addOptional(p, 'n_state_buf_len', default_n_state_buf_len, check_n_state_buf_len);
            addOptional(p, 'epsilon', default_epsilon, check_epsilon);

            parse(p, derivedArgs{:});


            obj.iteracts = 0;
            try
                % obj.n_win = p.Results.order;
                obj.mu = p.Results.mu;
                % obj.n_state_buf_len = p.Results.n_state_buf_len;
                obj.epsilon = p.Results.epsilon;

                % Inicializando a matriz H de pesos
                if (size(p.Results.H_ini, 1) ~= obj.y_dim) && (size(p.Results.H_ini, 2) ~= obj.x_dim)
                    error('The dimension of "H_ini" must be equal to ("y_dim", "x_dim").');
                end
                if p.Results.H_rnd_ini
                    obj.H = randn(default_y_dim, default_x_dim);
                else
                    obj.H = p.Results.H_ini;
                end

                % obj.state_buffer = zeros(obj.x_dim, obj.n_state_buf_len);
            catch exception
                error('An error occurred %s', exception.message);
            end

        end
        function y_hat_post = apply(obj, obs_buffer, state_buffer)

            obj.iteracts = obj.iteracts + 1;

            % n_win = 1 para LMS
            x = state_buffer(:,1);
            y = obs_buffer(:,1);

            % Expectation approximation
            % obj.R_hat = obj.expectation_approx(state_buffer, state_buffer, N);

            % obj.H = p_hat*(R_hat + obj.epsilon*eye(obj.x_dim)); Solução exata
            % w(n+1) = w(n) - \mu * (2 H*R - 2 p + 2 \epsilon H)
            % obj.H = obj.H - 2*obj.mu * (obj.H*obj.R_hat - obj.p_hat + obj.epsilon*obj.H);
            
            y_hat_prior = obj.H * x;
            % Error
            e = y - y_hat_prior;

            % Adapt
            if obj.epsilon == 0
                obj.H = obj.H + obj.mu * e * x';
            else % epsilon > 0
                obj.H = obj.H + obj.mu / (obj.epsilon + x'*x) * e * x';
            end
            
            y_hat_post = obj.H * x;
        end

        function H = get_H(obj)
            H = obj.H;
        end

        function obj = update_H(obj, H_new)
            obj.H = H_new;
        end

        function y_hat = get_y_hat(obj, st)
            y_hat = obj.H * st;
        end
    end
end