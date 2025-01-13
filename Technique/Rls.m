classdef Rls < Agent_technique
    %{ Rest Leasts Square (RLS) Algorithm
    %  Based on Adaptive Filtering Algorithms and Practical Implementation, Chap.5, Diniz 
    %}
    properties
        P               % matriz P RLS
        g             % vetor g RLS Alternativo
        alpha           % vetor e RLS Alternativo
        H               % matrix de pesos atual
        interacts       % número de iterações
        lambda          % lambda do algoritmo RLS
        delta           % fator de inicialização cega RLS
    end

    methods
        function obj = Rls(varargin)
            % Parametros da Classe derivada
            derivedParams = {'lambda', 'delta', 'H_ini', 'H_rnd_ini'};
            
            % Separação entre parâmetros da classe derivada e classe base
            isderivedParam = ismember(varargin(1:2:end), derivedParams);
            % para capturar tanto o arg_label, quanto o arg_value
            baseArgs = varargin(~reshape([isderivedParam; isderivedParam], 1, []));
            derivedArgs = varargin(reshape([isderivedParam; isderivedParam], 1, []));

            obj@Agent_technique(baseArgs{:});

            p = inputParser;

            default_lambda = 0.5;
            check_lambda = @(x) isnumeric(x) && isscalar(x) && (0 < x) && (x <= 1);

            default_delta = 0.1;
            check_delta = @(x) isnumeric(x) && isscalar(x) && (x > 0);

            default_H_ini = ones(obj.y_dim, obj.x_dim);
            check_H_ini = @(x) isnumeric(x);

            default_H_rnd_ini = false;
            check_H_rnd_ini = @(x) islogical(x);

            addOptional(p, 'lambda', default_lambda, check_lambda);
            addOptional(p, 'delta', default_delta, check_delta);
            addOptional(p, 'H_ini', default_H_ini, check_H_ini);
            addOptional(p, 'H_rnd_ini', default_H_rnd_ini, check_H_rnd_ini);

            parse(p, derivedArgs{:});

            obj.interacts = 0;
            try
                obj.lambda = p.Results.lambda;
                obj.delta = p.Results.delta;
                
                % Inicializando a matriz H de pesos
                if (size(p.Results.H_ini, 1) ~= obj.y_dim) && (size(p.Results.H_ini, 2) ~= obj.x_dim)
                    error('The dimension of "H_ini" must be equal to ("y_dim", "x_dim").');
                end
                if p.Results.H_rnd_ini
                    obj.H = randn(obj.y_dim, obj.x_dim);
                else
                    obj.H = p.Results.H_ini;
                end

                % Inicializando matriz P e vetor p_D
                obj.P = obj.delta * eye(obj.x_dim);
                obj.g = obj.P * zeros(obj.x_dim, 1);
                obj.alpha = zeros(obj.y_dim, 1);

            catch exception
                error('An error occurred %s', exception.message);
            end
        end
        function y_hat_post = apply(obj, obs_buffer, state_buffer)
            
            obj.interacts = obj.interacts + 1;

            % n_win = 1 para RLS
            x = state_buffer(:,1);
            y = obs_buffer(:,1);

            
            % Alternative RLS Algorithm %Ref Wikipedia
            y_hat_prior = obj.H * x;
            obj.alpha = y - y_hat_prior;
            obj.g = obj.P * x / (obj.lambda + x'*obj.P*x);

            obj.P = (1/obj.lambda)*obj.P - obj.g * x'*obj.P/obj.lambda;
            % Fator de correção de série divergente
            if norm(obj.P, 'fro') >= 1e15
                obj.reset_P();
                obj.update_lambda(0.001);
            end
            w = obj.H';
            w = w + obj.alpha*obj.g;
            obj.H = w';
            
            y_hat_post = obj.get_y_hat(x);
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
        function obj = reset_P(obj)
            obj.P = obj.delta * eye(obj.x_dim);
        end
        function obj = update_lambda(obj, add)
            if (0 < obj.lambda + add) && (obj.lambda + add <= 1)
                obj.lambda = obj.lambda + add;
            end
        end
        function obj = reset(obj)
            obj.H = zeros(size(obj.H));
            obj.P = obj.delta * eye(obj.x_dim);
            obj.g = obj.P * zeros(obj.x_dim, 1);
            obj.alpha = zeros(obj.y_dim, 1);
            obj.interacts = 0;
        end
    end
end