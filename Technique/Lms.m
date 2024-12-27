classdef Lms < Agent_technique
    properties
        R_hat               % estimado a partir do buffer de estados
        p_hat               % estimado a partir do buffer de estados e de observações
        H                   % matrix de pesos atual
        n_win               % tamanho da janela de observação
        mu                  % passo adaptativo
        iteracts            % número de iterações 
        % state_buffer        % buffer de estado
        % n_state_buf_len     % tamanho do buffer de estados
        epsilon             % termo de regularização
    end

    methods
        function obj = Lms(varargin)
            % disp(varargin)
            % Parametros da Classe derivada
            derivedParams = {'n_win', 'mu', 'H_ini', 'H_rnd_ini', 'epsilon'};

            % Separação entre parâmetros da classe derivada e base
            isderivedParam = ismember(varargin(1:2:end), derivedParams);
            % para capturar tanto o arg_label, quanto o arg_value
            baseArgs = varargin(~reshape([isderivedParam; isderivedParam], 1, []));
            derivedArgs = varargin(reshape([isderivedParam; isderivedParam], 1, []));
            % disp(baseArgs)

            obj@Agent_technique(baseArgs{:});
           
            p = inputParser;

            default_n_win = 20;
            check_n_win = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);

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

            addOptional(p, 'n_win', default_n_win, check_n_win);
            addOptional(p, 'mu', default_mu, check_mu);
            addOptional(p, 'H_ini', default_H_ini, check_H_ini);
            addOptional(p, 'H_rnd_ini', default_H_rnd_ini, check_H_rnd_ini);
            % addOptional(p, 'n_state_buf_len', default_n_state_buf_len, check_n_state_buf_len);
            addOptional(p, 'epsilon', default_epsilon, check_epsilon);

            parse(p, derivedArgs{:});


            obj.iteracts = 0;
            try
                obj.n_win = p.Results.n_win;
                obj.mu = p.Results.mu;
                % obj.n_state_buf_len = p.Results.n_state_buf_len;
                obj.epsilon = p.Results.epsilon;

                obj.R_hat = zeros(obj.x_dim, obj.x_dim);
                obj.p_hat = zeros(obj.y_dim, obj.x_dim);

                % Inicializando a matriz H de pesos
                if (size(p.Results.H_ini, 1) ~= obj.y_dim) && (size(p.Results.x_ini, 2) ~= obj.x_dim)
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

            obj.iteracts = 0;

        end
        function out = apply(obj, obs_buffer, state_buffer)

            obj.iteracts = obj.iteracts + 1;

            % Define o tamanho da Janela a ser utilizada
            if obj.iteracts < obj.n_win
                N = obj.iteracts;
            else
                N = obj.n_win;
            end

            % Expectation approximation
            obj.R_hat = obj.expectation_approx(state_buffer, state_buffer, N);
            obj.p_hat = obj.expectation_approx(obs_buffer, state_buffer, N);

            % obj.H = p_hat*(R_hat + obj.epsilon*eye(obj.x_dim)); Solução exata
            % w(n+1) = w(n) - \mu * (2 H*R - 2 p + 2 \epsilon H)
            obj.H = obj.H - 2*obj.mu * (obj.H*obj.R_hat - obj.p_hat + obj.epsilon*obj.H);

            out = obj.H * state_buffer(:,1);
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
    methods (Static)
        function out = expectation_approx(buff1, buff2, N)
            out = zeros(size(buff1,1), size(buff2,1));
            for i = 1:N
                out = buff1(:,i) * buff2(:,i)' + out;
            end
            out = out / N;
        end
    end
end